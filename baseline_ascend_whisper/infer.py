"""推理脚本：生成提交 CSV。支持 TTA、多检查点集成、分数校准、多卡分布式推理。"""
from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import time

import numpy as np
import torch
import whisper
from tqdm import tqdm
from whisper.audio import mel_filters

from config import AUDIO, PATHS, TRAIN
from data import batch_read_pairs, load_pairs, preload_zip
from distributed_utils import (
    cleanup_distributed,
    create_tp_dp_groups,
    setup_distributed,
    shard_state_dict,
)
from model import SiameseKWS


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(PATHS.ckpt_dir, "best.pt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "submission.csv"))
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--tta", type=int, default=0,
                    help="TTA 次数 (0=不使用TTA, 推荐3-5)")
    ap.add_argument("--ensemble", action="store_true",
                    help="使用 top1~top3 检查点集成")
    ap.add_argument("--calibrate", action="store_true",
                    help="使用 dev 集做 Platt 校准")
    ap.add_argument("--tp", type=int, default=1, help="张量并行度")
    ap.add_argument("--dp", type=int, default=1, help="数据并行度")
    ap.add_argument("--nproc_per_node", type=int, default=1, help="使用的NPU卡数")
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


def tta_augment_mel(mel: torch.Tensor, n_aug: int) -> list:
    B, n_mels, T = mel.shape
    augs = [mel]
    for _ in range(n_aug):
        aug = mel.clone()
        f_mask = torch.randint(1, 4, (1,)).item()
        f0 = torch.randint(0, max(n_mels - f_mask, 1), (1,)).item()
        aug[:, f0:f0 + f_mask, :] = 0
        t_mask = torch.randint(5, 30, (1,)).item()
        t0 = torch.randint(0, max(T - t_mask, 1), (1,)).item()
        aug[:, :, t0:t0 + t_mask] = 0
        augs.append(aug)
    return augs


def load_model_from_ckpt(ckpt_path: str, device: str,
                         tp_rank: int = 0, tp_world_size: int = 1,
                         tp_group=None) -> SiameseKWS:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    whisper_model = ckpt.get("whisper_model", AUDIO.whisper_model_name)
    embed_dim = ckpt.get("embed_dim", TRAIN.embed_dim)
    use_mlp = ckpt.get("use_mlp", False)
    model = SiameseKWS(whisper_model, embed_dim, device=device, use_mlp=use_mlp,
                       tp_rank=tp_rank, tp_world_size=tp_world_size, tp_group=tp_group)

    full_state = ckpt["model"]
    if tp_world_size > 1:
        sharded_state = shard_state_dict(full_state, model, tp_rank, tp_world_size)
        model.load_state_dict(sharded_state, strict=False)
    else:
        model.load_state_dict(full_state, strict=False)

    model.to(device).eval()
    print(f"loaded {ckpt_path} (dev mean AUC={ckpt.get('auc')}, epoch={ckpt.get('epoch')})")
    return model


@torch.no_grad()
def predict_single(model, zip_path, csv_path, prefix, device, bs, n_tta=0,
                   dp_rank=0, dp=1, dp_group=None):
    preload_zip(zip_path)
    pairs = load_pairs(csv_path, False)
    n = len(pairs)

    if dp > 1:
        chunk_size = (n + dp - 1) // dp
        start_idx = dp_rank * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        dp_pairs = pairs[start_idx:end_idx]
    else:
        dp_pairs = pairs
        start_idx = 0

    n_local = len(dp_pairs)

    model.eval()
    all_logits = None
    local_ids = []
    t0 = time.time()

    n_batches = (n_local + bs - 1) // bs
    pbar = tqdm(range(n_batches), desc=f"infer/{prefix}", dynamic_ncols=True,
                unit="batch", leave=True)
    for bi in pbar:
        start = bi * bs
        end = min(start + bs, n_local)
        batch_pairs = dp_pairs[start:end]

        e, q, _, ids = batch_read_pairs(batch_pairs, zip_path, AUDIO, inference=True)
        e_mel = npu_batch_mel(e, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)
        q_mel = npu_batch_mel(q, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)

        if n_tta > 0:
            e_augs = tta_augment_mel(e_mel, n_tta)
            q_augs = tta_augment_mel(q_mel, n_tta)
            logit_sum = None
            count = 0
            for e_aug in e_augs:
                for q_aug in q_augs:
                    with torch.npu.amp.autocast(enabled=True):
                        logit = model(e_aug, q_aug)
                    if logit_sum is None:
                        logit_sum = torch.zeros_like(logit)
                    logit_sum += logit
                    count += 1
            logit = logit_sum / count
        else:
            with torch.npu.amp.autocast(enabled=True):
                logit = model(e_mel, q_mel)

        if all_logits is None:
            all_logits = logit.float().cpu().numpy()
        else:
            all_logits = np.concatenate([all_logits, logit.float().cpu().numpy()])

        local_ids.extend(ids)

        cur_count = len(local_ids)
        elapsed = time.time() - t0
        speed = cur_count / elapsed if elapsed > 0 else 0
        pbar.set_postfix({"speed": f"{speed:.0f}sa/s"})

    if dp > 1 and dp_group is not None:
        import torch.distributed as dist
        global_rank = dist.get_rank()
        tmp_dir = os.path.join("/tmp", "infer_dp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_file = os.path.join(tmp_dir, f"rank{global_rank}.pkl")
        with open(tmp_file, "wb") as f:
            pickle.dump({"logits": all_logits, "ids": local_ids}, f)
        dist.barrier(group=dp_group)

        if dp_rank == 0:
            logits_list = []
            ids_list = []
            for d in range(dp):
                r = d * tp + tp_rank
                f_path = os.path.join(tmp_dir, f"rank{r}.pkl")
                with open(f_path, "rb") as f:
                    data = pickle.load(f)
                logits_list.append(data["logits"])
                ids_list.extend(data["ids"])
                os.remove(f_path)
            all_logits = np.concatenate(logits_list)
            local_ids = ids_list

    elapsed = time.time() - t0
    if dp_rank == 0 or dp == 1:
        print(f"[{prefix}] done: {len(local_ids)} samples, {elapsed:.1f}s, "
              f"speed={len(local_ids)/elapsed:.0f} sa/s")

    rows = [(f"{prefix}_{pid}", 0.0) for pid in local_ids]
    return rows, all_logits


@torch.no_grad()
def calibrate_on_dev(model, device, dp_rank=0, dp=1, dp_group=None, is_main=False):
    from sklearn.metrics import roc_auc_score
    from sklearn.isotonic import IsotonicRegression

    if is_main:
        print("[calibrate] computing dev predictions for calibration ...")
    preload_zip(PATHS.dev_seen_zip)
    preload_zip(PATHS.dev_unseen_zip)
    dev_seen_pairs = load_pairs(PATHS.dev_seen_csv, True)
    dev_unseen_pairs = load_pairs(PATHS.dev_unseen_csv, True)

    all_probs = []
    all_labels = []

    for pairs, zip_path, desc in [
        (dev_seen_pairs, PATHS.dev_seen_zip, "cal_seen"),
        (dev_unseen_pairs, PATHS.dev_unseen_zip, "cal_unseen"),
    ]:
        n = len(pairs)
        bs = 256
        for start in tqdm(range(0, n, bs), desc=desc, leave=False, disable=not is_main):
            end = min(start + bs, n)
            batch_pairs = pairs[start:end]
            e, q, y, _ = batch_read_pairs(batch_pairs, zip_path, AUDIO, inference=False)
            e_mel = npu_batch_mel(e, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)
            q_mel = npu_batch_mel(q, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)
            with torch.npu.amp.autocast(enabled=True):
                logit = model(e_mel, q_mel)
            if is_main:
                prob = torch.sigmoid(logit.float()).cpu().numpy()
                all_probs.append(prob)
                all_labels.append(y.numpy())

    if is_main:
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        print(f"[calibrate] dev AUC before calibration: {roc_auc_score(labels, probs):.4f}")

        iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso_reg.fit(probs, labels)
        cal_probs = iso_reg.predict(probs)
        print(f"[calibrate] dev AUC after calibration: {roc_auc_score(labels, cal_probs):.4f}")
        return iso_reg
    return None


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


def worker_main(rank, args, tp, dp, world_size):
    is_distributed = world_size > 1

    if is_distributed:
        setup_distributed(rank, world_size, backend="hccl")
        tp_group, dp_group, tp_rank, dp_rank = create_tp_dp_groups(tp, dp)
    else:
        tp_group, dp_group, tp_rank, dp_rank = None, None, 0, 0

    is_main = (rank == 0)

    device = f"npu:{rank}" if torch.npu.is_available() else "cpu"
    if device.startswith("npu"):
        torch.npu.set_device(rank)

    iso_reg = None

    if args.ensemble:
        ckpt_paths = []
        for r in range(1, 4):
            p = os.path.join(PATHS.ckpt_dir, f"top{r}.pt")
            if os.path.exists(p):
                ckpt_paths.append(p)
        if not ckpt_paths:
            ckpt_paths = [args.ckpt]
        if is_main:
            print(f"[ensemble] using {len(ckpt_paths)} checkpoints: {ckpt_paths}")

        models = []
        for cp in ckpt_paths:
            m = load_model_from_ckpt(cp, device, tp_rank=tp_rank,
                                     tp_world_size=tp, tp_group=tp_group)
            warmup(m, device)
            models.append(m)

        if args.calibrate:
            iso_reg = calibrate_on_dev(models[0], device, dp_rank=dp_rank,
                                       dp=dp, dp_group=dp_group, is_main=is_main)

        all_rows = []
        for prefix, zip_path, csv_path in [
            ("seen", PATHS.eval_seen_zip, PATHS.eval_seen_csv),
            ("unseen", PATHS.eval_unseen_zip, PATHS.eval_unseen_csv),
        ]:
            preload_zip(zip_path)
            pairs = load_pairs(csv_path, False)
            n = len(pairs)
            bs = args.bs

            if dp > 1:
                chunk_size = (n + dp - 1) // dp
                start_idx = dp_rank * chunk_size
                end_idx = min(start_idx + chunk_size, n)
                dp_pairs = pairs[start_idx:end_idx]
            else:
                dp_pairs = pairs

            n_local = len(dp_pairs)
            logits_accum = np.zeros(n_local, dtype=np.float64)

            for mi, model in enumerate(models):
                model.eval()
                model_logits = []
                pbar = tqdm(range(0, n_local, bs), desc=f"ensemble/{prefix}/m{mi}",
                            dynamic_ncols=True, leave=False, disable=not is_main)
                for start in pbar:
                    end = min(start + bs, n_local)
                    batch_pairs = dp_pairs[start:end]
                    e, q, _, ids = batch_read_pairs(batch_pairs, zip_path, AUDIO, inference=True)
                    e_mel = npu_batch_mel(e, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)
                    q_mel = npu_batch_mel(q, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)
                    with torch.npu.amp.autocast(enabled=True):
                        logit = model(e_mel, q_mel)
                    model_logits.append(logit.float().cpu().numpy())
                logits_m = np.concatenate(model_logits)
                logits_accum += logits_m

            logits_accum /= len(models)

            if dp > 1 and dp_group is not None:
                import torch.distributed as dist
                global_rank = dist.get_rank()
                tmp_dir = os.path.join("/tmp", "infer_ensemble_dp")
                os.makedirs(tmp_dir, exist_ok=True)
                tmp_file = os.path.join(tmp_dir, f"rank{global_rank}_{prefix}.pkl")
                with open(tmp_file, "wb") as f:
                    pickle.dump({"logits": logits_accum, "pairs": dp_pairs}, f)
                dist.barrier(group=dp_group)

                if dp_rank == 0:
                    all_logits_list = []
                    all_dp_pairs = []
                    for d in range(dp):
                        r = d * tp + tp_rank
                        f_path = os.path.join(tmp_dir, f"rank{r}_{prefix}.pkl")
                        with open(f_path, "rb") as f:
                            data = pickle.load(f)
                        all_logits_list.append(data["logits"])
                        all_dp_pairs.extend(data["pairs"])
                        os.remove(f_path)
                    logits_accum = np.concatenate(all_logits_list)
                    dp_pairs = all_dp_pairs

            if dp_rank == 0 or dp == 1:
                probs = 1.0 / (1.0 + np.exp(-logits_accum))
                if iso_reg is not None:
                    probs = iso_reg.predict(probs)
                for i, p in enumerate(dp_pairs):
                    all_rows.append((f"{prefix}_{p['id']}", float(probs[i])))
                print(f"[{prefix}] ensemble done: {len(dp_pairs)} samples, "
                      f"mean={probs.mean():.4f}, std={probs.std():.4f}")

    else:
        model = load_model_from_ckpt(args.ckpt, device, tp_rank=tp_rank,
                                     tp_world_size=tp, tp_group=tp_group)
        warmup(model, device)

        if args.calibrate:
            iso_reg = calibrate_on_dev(model, device, dp_rank=dp_rank,
                                       dp=dp, dp_group=dp_group, is_main=is_main)

        all_rows = []
        for prefix, zip_path, csv_path in [
            ("seen", PATHS.eval_seen_zip, PATHS.eval_seen_csv),
            ("unseen", PATHS.eval_unseen_zip, PATHS.eval_unseen_csv),
        ]:
            rows, logits = predict_single(model, zip_path, csv_path, prefix,
                                          device, args.bs, n_tta=args.tta,
                                          dp_rank=dp_rank, dp=dp, dp_group=dp_group)
            if dp_rank == 0 or dp == 1:
                probs = 1.0 / (1.0 + np.exp(-logits))
                if iso_reg is not None:
                    probs = iso_reg.predict(probs)
                for i, (row_id, _) in enumerate(rows):
                    all_rows.append((row_id, float(probs[i])))

    if is_main:
        print(f"total: {len(all_rows)} rows")
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "posterior"])
            w.writerows(all_rows)
        print(f"wrote {args.out}")

    if is_distributed:
        cleanup_distributed()


def main():
    args = parse_args()

    if not torch.npu.is_available():
        raise SystemExit("NPU 不可用")
    try:
        import tbe  # noqa: F401
    except Exception as exc:
        raise SystemExit("NPU 环境缺少 tbe 模块") from exc

    tp = args.tp
    dp = args.dp
    world_size = args.nproc_per_node

    assert tp * dp == world_size, \
        f"tp({tp}) * dp({dp}) = {tp*dp} != nproc_per_node({world_size})"

    if world_size == 1:
        os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
        if "ASCEND_GLOBAL_LOG_LEVEL" not in os.environ:
            os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
        worker_main(0, args, tp=1, dp=1, world_size=1)
    else:
        os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES",
                              ",".join(str(i) for i in range(world_size)))
        if "ASCEND_GLOBAL_LOG_LEVEL" not in os.environ:
            os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
        torch.multiprocessing.spawn(
            worker_main,
            args=(args, tp, dp, world_size),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
