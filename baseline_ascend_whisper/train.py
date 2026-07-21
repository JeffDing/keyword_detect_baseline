from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import math
import time

import numpy as np
import torch
import whisper
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from whisper.audio import mel_filters

from config import AUDIO, PATHS, TRAIN
from data import batch_read_pairs, load_pairs, load_similar_pairs, preload_zip
from model import SiameseKWS


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=TRAIN.epochs)
    ap.add_argument("--bs", type=int, default=TRAIN.batch_size)
    ap.add_argument("--lr", type=float, default=TRAIN.lr)
    ap.add_argument("--subset", type=int, default=TRAIN.train_subset,
                    help="训练子集大小，越小分数通常越低")
    ap.add_argument("--hard_mining_every", type=int, default=TRAIN.hard_mining_every)
    ap.add_argument("--hard_top_k", type=int, default=TRAIN.hard_top_k)
    ap.add_argument("--hard_weight", type=float, default=TRAIN.hard_weight)
    ap.add_argument("--embed_dim", type=int, default=TRAIN.embed_dim)
    ap.add_argument("--use_mlp", action="store_true", default=TRAIN.use_mlp)
    ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--warmup_epochs", type=int, default=TRAIN.warmup_epochs)
    ap.add_argument("--save_top_k", type=int, default=TRAIN.save_top_k)
    ap.add_argument("--spec_augment", action="store_true", default=TRAIN.spec_augment)
    ap.add_argument("--no_spec_augment", action="store_true")
    ap.add_argument("--out", type=str, default=os.path.join(PATHS.ckpt_dir, "best.pt"))
    return ap.parse_args()


def spec_augment(mel: torch.Tensor, freq_mask: int = 2, time_mask: int = 10) -> torch.Tensor:
    B, n_mels, T = mel.shape
    device = mel.device
    augmented = mel.clone()
    for _ in range(freq_mask):
        f = torch.randint(0, max(freq_mask, 1), (1,)).item()
        f0 = torch.randint(0, max(n_mels - f, 1), (1,)).item()
        augmented[:, f0:f0 + f, :] = 0
    for _ in range(time_mask):
        t = torch.randint(0, max(time_mask, 1), (1,)).item()
        t0 = torch.randint(0, max(T - t, 1), (1,)).item()
        augmented[:, :, t0:t0 + t] = 0
    return augmented


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
def evaluate(model, pairs, zip_path, device, desc="eval"):
    model.eval()
    probs, labels = [], []
    n = len(pairs)
    bs = 256
    pbar = tqdm(range(0, n, bs), desc=desc, dynamic_ncols=True, leave=False)
    for start in pbar:
        end = min(start + bs, n)
        batch_pairs = pairs[start:end]
        e, q, y, _ = batch_read_pairs(batch_pairs, zip_path, AUDIO, inference=False)
        e_mel = npu_batch_mel(e, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)
        q_mel = npu_batch_mel(q, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)
        with torch.npu.amp.autocast(enabled=True):
            logit = model(e_mel, q_mel)
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        labels.append(y.numpy())
    return roc_auc_score(np.concatenate(labels), np.concatenate(probs))


def _init_npu_env():
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
    if "ASCEND_GLOBAL_LOG_LEVEL" not in os.environ:
        os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
    torch.npu.set_device(0)


def mine_batch_hard_negatives(model, e, q, y, top_k=1):
    B = e.size(0)
    with torch.no_grad():
        e_feat, q_feat = model.get_embeddings(e, q)
        e_g = e_feat.mean(dim=1)
        q_g = q_feat.mean(dim=1)
        sim = torch.matmul(e_g, q_g.t())

    sim_device = sim.device
    pos_indices = (y.to(sim_device) == 1).nonzero(as_tuple=True)[0]
    neg_indices = (y.to(sim_device) == 0).nonzero(as_tuple=True)[0]

    hard_e = []
    hard_q = []
    hard_y = []

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return None, None, None

    for i in pos_indices:
        neg_scores = sim[i, neg_indices]
        k = min(top_k, len(neg_indices))
        top_j = neg_indices[torch.topk(neg_scores, k).indices[0]]
        hard_e.append(e[top_j:top_j + 1])
        hard_q.append(q[i:i + 1])
        hard_y.append(torch.tensor([0.0], device=e.device))

    return torch.cat(hard_e), torch.cat(hard_q), torch.cat(hard_y)


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, current_step: int):
        if current_step < self.warmup_steps:
            scale = current_step / max(self.warmup_steps, 1)
        else:
            progress = (current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale


def main():
    args = parse_args()
    if args.no_mlp:
        args.use_mlp = False
    if args.no_spec_augment:
        args.spec_augment = False

    torch.manual_seed(TRAIN.seed)
    np.random.seed(TRAIN.seed)
    device = "npu" if torch.npu.is_available() else "cpu"
    if device == "npu":
        _init_npu_env()
    os.makedirs(PATHS.ckpt_dir, exist_ok=True)

    all_pairs = load_pairs(PATHS.train_csv, with_label=True)
    similar_pairs = load_similar_pairs(PATHS.similar_word_csv)
    all_pairs.extend(similar_pairs)

    n = min(args.subset, len(all_pairs))
    idx = np.random.default_rng(TRAIN.seed).permutation(len(all_pairs))[:n]
    train_pairs = [all_pairs[i] for i in idx]
    print(f"train: {n} / {len(all_pairs)} pairs (incl. similar={len(similar_pairs)})")
    print(f"config: embed_dim={args.embed_dim}, use_mlp={args.use_mlp}, "
          f"spec_augment={args.spec_augment}, warmup={args.warmup_epochs}")

    preload_zip(PATHS.train_zip)
    preload_zip(PATHS.dev_seen_zip)
    preload_zip(PATHS.dev_unseen_zip)

    dev_seen_pairs = load_pairs(PATHS.dev_seen_csv, True)
    dev_unseen_pairs = load_pairs(PATHS.dev_unseen_csv, True)

    model = SiameseKWS(AUDIO.whisper_model_name, args.embed_dim,
                       device=device, use_mlp=args.use_mlp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {n_params:,} ({n_params/1e6:.2f}M), trainable: {n_trainable:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(TRAIN.pos_weight, device=device))

    n_batches = n // args.bs
    total_steps = args.epochs * n_batches
    warmup_steps = args.warmup_epochs * n_batches
    scheduler = CosineWarmupScheduler(opt, warmup_steps, total_steps)

    best = 0.0
    topk_ckpts = []
    global_step = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0

        ep_idx = np.random.permutation(n)
        ep_pairs = [train_pairs[i] for i in ep_idx]

        pbar = tqdm(range(n_batches), desc=f"ep{ep}", dynamic_ncols=True)
        for it in pbar:
            start = it * args.bs
            end = start + args.bs
            batch_pairs = ep_pairs[start:end]

            e, q, y, _ = batch_read_pairs(batch_pairs, PATHS.train_zip, AUDIO, inference=False)
            e_mel = npu_batch_mel(e, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)
            q_mel = npu_batch_mel(q, device, n_mels=AUDIO.n_mels, max_frames=AUDIO.max_frames)

            if args.spec_augment and model.training:
                e_mel = spec_augment(e_mel, TRAIN.spec_aug_freq_mask, TRAIN.spec_aug_time_mask)
                q_mel = spec_augment(q_mel, TRAIN.spec_aug_freq_mask, TRAIN.spec_aug_time_mask)

            y = y.to(device)

            opt.zero_grad()
            with torch.npu.amp.autocast(enabled=True):
                logits = model(e_mel, q_mel)
                loss = crit(logits, y)

                if args.hard_top_k > 0 and ep % args.hard_mining_every == 0:
                    hard_e, hard_q, hard_y = mine_batch_hard_negatives(
                        model, e_mel, q_mel, y, top_k=args.hard_top_k)
                    if hard_e is not None:
                        hard_logits = model(hard_e, hard_q)
                        loss_hard = crit(hard_logits, hard_y)
                        loss = loss + args.hard_weight * loss_hard

            loss.backward()
            opt.step()
            scheduler.step(global_step)
            global_step += 1
            loss_sum += loss.item()
            pbar.set_postfix({"loss": f"{loss_sum/(it+1):.4f}",
                              "lr": f"{opt.param_groups[0]['lr']:.2e}"})

        auc_s = evaluate(model, dev_seen_pairs, PATHS.dev_seen_zip, device, desc="seen")
        auc_u = evaluate(model, dev_unseen_pairs, PATHS.dev_unseen_zip, device, desc="unseen")
        mean = (auc_s + auc_u) / 2
        print(f"[epoch {ep}] seen={auc_s:.4f} unseen={auc_u:.4f} "
              f"mean={mean:.4f} ({time.time()-t0:.0f}s)")

        ckpt_data = {
            "model": model.state_dict(),
            "embed_dim": args.embed_dim,
            "whisper_model": AUDIO.whisper_model_name,
            "use_mlp": args.use_mlp,
            "auc": mean,
            "epoch": ep,
        }

        if mean > best:
            best = mean
            torch.save(ckpt_data, args.out)
            print(f"  saved best -> {args.out}")

        topk_ckpts.append((mean, ep, ckpt_data))
        topk_ckpts.sort(key=lambda x: x[0], reverse=True)
        topk_ckpts = topk_ckpts[:args.save_top_k]
        for rank, (auc_val, ep_val, data) in enumerate(topk_ckpts):
            path = os.path.join(PATHS.ckpt_dir, f"top{rank+1}.pt")
            torch.save(data, path)

    print(f"done. best dev mean AUC = {best:.4f}")
    print(f"top-{args.save_top_k} checkpoints saved in {PATHS.ckpt_dir}")


if __name__ == "__main__":
    main()
