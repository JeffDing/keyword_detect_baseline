"""推理脚本：生成提交 CSV。"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import whisper
from tqdm import tqdm
from whisper.audio import mel_filters

from config import AUDIO, PATHS, TRAIN
from data import batch_read_pairs, load_pairs, preload_zip
from model import SiameseKWS


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(PATHS.ckpt_dir, "best.pt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "submission.csv"))
    ap.add_argument("--bs", type=int, default=256)
    return ap.parse_args()


def npu_batch_mel(wavs_cpu: torch.Tensor, device: str,
                  n_mels: int = 80, max_frames: int = 3000,
                  sub_bs: int = 32) -> torch.Tensor:
    mels = []
    mel_fb = mel_filters(torch.device(device), n_mels)
    for i in range(0, wavs_cpu.size(0), sub_bs):
        wav_sub = wavs_cpu[i:i + sub_bs].to(device)
        window = torch.hann_window(400, device=device, dtype=wav_sub.dtype)
        spec = torch.stft(
            wav_sub, n_fft=400, hop_length=160, win_length=400,
            window=window, center=True, pad_mode="reflect",
            normalized=False, onesided=True, return_complex=True,
        )
        magnitudes = spec.real.square() + spec.imag.square()
        magnitudes = magnitudes[..., :-1]
        mel_spec = torch.matmul(mel_fb, magnitudes)
        log_spec = torch.clamp(mel_spec, min=1e-10).log() / math.log(10)
        max_vals = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, max_vals - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        log_spec = log_spec[:, :, :max_frames]
        if log_spec.size(2) < max_frames:
            log_spec = torch.nn.functional.pad(log_spec, (0, max_frames - log_spec.size(2)))
        mels.append(log_spec)
    return torch.cat(mels, dim=0)


@torch.no_grad()
def predict(model, zip_path, csv_path, prefix, device, bs):
    preload_zip(zip_path)
    pairs = load_pairs(csv_path, False)
    n = len(pairs)

    model.eval()
    all_probs = []
    rows = []
    t0 = time.time()

    n_batches = (n + bs - 1) // bs
    pbar = tqdm(range(n_batches), desc=f"infer/{prefix}", dynamic_ncols=True,
                unit="batch", leave=True)
    for bi in pbar:
        start = bi * bs
        end = min(start + bs, n)
        batch_pairs = pairs[start:end]

        e, q, _, ids = batch_read_pairs(batch_pairs, zip_path, AUDIO, inference=True)

        e_mel = npu_batch_mel(e, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)
        q_mel = npu_batch_mel(q, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)

        with torch.npu.amp.autocast(enabled=True):
            logit = model(e_mel, q_mel)
        prob = torch.sigmoid(logit.float()).cpu().numpy()
        all_probs.append(prob)
        for pid, p in zip(ids, prob):
            rows.append((f"{prefix}_{pid}", float(p)))

        cur_probs = np.concatenate(all_probs)
        elapsed = time.time() - t0
        speed = len(cur_probs) / elapsed if elapsed > 0 else 0
        pbar.set_postfix({
            "speed": f"{speed:.0f}sa/s",
            "mean": f"{cur_probs.mean():.4f}",
        })

    elapsed = time.time() - t0
    cur_probs = np.concatenate(all_probs)
    print(f"[{prefix}] done: {len(rows)} samples, {elapsed:.1f}s, "
          f"speed={len(rows)/elapsed:.0f} sa/s, "
          f"mean={cur_probs.mean():.4f}, std={cur_probs.std():.4f}")
    return rows


def _init_npu_env():
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        count = torch.npu.device_count()
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(count))
    if "ASCEND_GLOBAL_LOG_LEVEL" not in os.environ:
        os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
    torch.npu.set_device(0)


def warmup(model, device):
    print("[warmup] running dummy forward on NPU ...")
    torch.manual_seed(42)
    dummy_wav = torch.randn(2, 480000)
    e_mel = npu_batch_mel(dummy_wav, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)
    q_mel = npu_batch_mel(dummy_wav, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)
    t0 = time.time()
    with torch.no_grad():
        with torch.npu.amp.autocast(enabled=True):
            out = model(e_mel, q_mel)
    torch.npu.synchronize()
    print(f"[warmup] done in {time.time()-t0:.1f}s, output shape={out.shape}, device={out.device}")


def main():
    args = parse_args()
    if not torch.npu.is_available():
        raise SystemExit("NPU 不可用")
    try:
        import tbe  # noqa: F401
    except Exception as exc:
        raise SystemExit("NPU 环境缺少 tbe 模块") from exc

    device = "npu"
    _init_npu_env()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    whisper_model = ckpt.get("whisper_model", AUDIO.whisper_model_name)
    model = SiameseKWS(whisper_model, ckpt.get("embed_dim", TRAIN.embed_dim), device=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    print(f"loaded {args.ckpt} (dev mean AUC={ckpt.get('auc')})")

    warmup(model, device)

    rows = predict(model, PATHS.eval_seen_zip, PATHS.eval_seen_csv, "seen", device, args.bs)
    rows += predict(model, PATHS.eval_unseen_zip, PATHS.eval_unseen_csv, "unseen", device, args.bs)
    print(f"total: {len(rows)} rows")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "posterior"])
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
