from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import AUDIO, PATHS, TRAIN
from data import PairDataset, collate, load_pairs, load_similar_pairs
from model import SiameseKWS


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=TRAIN.epochs)
    ap.add_argument("--bs", type=int, default=TRAIN.batch_size)
    ap.add_argument("--lr", type=float, default=TRAIN.lr)
    ap.add_argument("--subset", type=int, default=TRAIN.train_subset,
                    help="训练子集大小，越小分数通常越低")
    ap.add_argument("--workers", type=int, default=TRAIN.num_workers)
    ap.add_argument("--hard_mining_every", type=int, default=TRAIN.hard_mining_every)
    ap.add_argument("--hard_top_k", type=int, default=TRAIN.hard_top_k)
    ap.add_argument("--hard_weight", type=float, default=TRAIN.hard_weight)
    ap.add_argument("--out", type=str, default=os.path.join(PATHS.ckpt_dir, "best.pt"))
    ap.add_argument("--persistent_workers", type=lambda x: x.lower() == "true",
                    default=TRAIN.persistent_workers)
    ap.add_argument("--prefetch_factor", type=int, default=TRAIN.prefetch_factor)
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, desc="eval"):
    model.eval()
    probs, labels = [], []
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
    for e, q, y, _ in pbar:
        e, q = e.to(device), q.to(device)
        logit = model(e, q)
        probs.append(torch.sigmoid(logit).cpu().numpy())
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


def main():
    args = parse_args()
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

    train_ds = PairDataset(train_pairs, PATHS.train_zip, AUDIO)
    persistent_workers = args.persistent_workers and args.workers > 0
    loader_kwargs = dict(
        num_workers=args.workers,
        collate_fn=collate,
        persistent_workers=persistent_workers,
    )
    if persistent_workers:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              drop_last=True, **loader_kwargs)

    def dev_loader(zip_p, csv_p):
        ds = PairDataset(load_pairs(csv_p, True), zip_p, AUDIO)
        return DataLoader(ds, batch_size=args.bs, shuffle=False, **loader_kwargs)

    dev_seen = dev_loader(PATHS.dev_seen_zip, PATHS.dev_seen_csv)
    dev_unseen = dev_loader(PATHS.dev_unseen_zip, PATHS.dev_unseen_csv)

    model = SiameseKWS(AUDIO.whisper_model_name, TRAIN.embed_dim, device=device).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"[train] device={device}, torch.npu.current_device()={torch.npu.current_device()}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(TRAIN.pos_weight, device=device))

    # debug_printed = False
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"ep{ep}", dynamic_ncols=True)
        for it, (e, q, y, _) in enumerate(pbar, 1):
            e, q, y = e.to(device), q.to(device), y.to(device)
            # if not debug_printed:
            #     print(f"[debug] e.device={e.device}, q.device={q.device}, y.device={y.device}")
            #     for name, p in model.named_parameters():
            #         print(f"[debug] param {name}: device={p.device}")
            #         break
            #     debug_printed = True
            opt.zero_grad()
            logits = model(e, q)
            loss = crit(logits, y)

            if args.hard_top_k > 0 and ep % args.hard_mining_every == 0:
                hard_e, hard_q, hard_y = mine_batch_hard_negatives(
                    model, e, q, y, top_k=args.hard_top_k)
                if hard_e is not None:
                    hard_logits = model(hard_e, hard_q)
                    loss_hard = crit(hard_logits, hard_y)
                    loss = loss + args.hard_weight * loss_hard

            loss.backward()
            opt.step()
            loss_sum += loss.item()
            pbar.set_postfix({"loss": f"{loss_sum/it:.4f}"})

        auc_s = evaluate(model, dev_seen, device, desc="seen")
        auc_u = evaluate(model, dev_unseen, device, desc="unseen")
        mean = (auc_s + auc_u) / 2
        print(f"[epoch {ep}] seen={auc_s:.4f} unseen={auc_u:.4f} "
              f"mean={mean:.4f} ({time.time()-t0:.0f}s)")

        if mean > best:
            best = mean
            torch.save({"model": model.state_dict(),
                        "embed_dim": TRAIN.embed_dim,
                        "whisper_model": AUDIO.whisper_model_name,
                        "auc": mean}, args.out)
            print(f"  saved -> {args.out}")

    print(f"done. best dev mean AUC = {best:.4f}")


if __name__ == "__main__":
    main()
