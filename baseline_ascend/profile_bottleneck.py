"""分析 baseline_ascend 训练瓶颈。"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import AUDIO, PATHS, TRAIN
from data import PairDataset, collate, load_pairs
from model import SiameseKWS


def _init_npu_env():
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
    if "ASCEND_GLOBAL_LOG_LEVEL" not in os.environ:
        os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
    torch.npu.set_device(0)


def main():
    _init_npu_env()
    device = "npu"

    all_pairs = load_pairs(PATHS.train_csv, with_label=True)
    n = min(2000, len(all_pairs))
    idx = np.random.default_rng(TRAIN.seed).permutation(len(all_pairs))[:n]
    train_pairs = [all_pairs[i] for i in idx]

    train_ds = PairDataset(train_pairs, PATHS.train_zip, AUDIO)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=0, collate_fn=collate,
                              drop_last=True)

    model = SiameseKWS(AUDIO.n_mels, TRAIN.embed_dim)
    ckpt_path = os.path.join(PATHS.ckpt_dir, "best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(TRAIN.pos_weight, device=device))

    batches = 20
    times = {"total": [], "data_h2d": [], "forward": [], "backward": [], "step": [], "item": []}

    loader_iter = iter(train_loader)
    for i in range(batches):
        e, q, y, _ = next(loader_iter)

        t0 = time.perf_counter()
        e = e.to(device, non_blocking=False)
        q = q.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)
        torch.npu.synchronize()
        t1 = time.perf_counter()

        opt.zero_grad()
        logit = model(e, q)
        torch.npu.synchronize()
        t2 = time.perf_counter()

        loss = crit(logit, y)
        loss.backward()
        torch.npu.synchronize()
        t3 = time.perf_counter()

        opt.step()
        torch.npu.synchronize()
        t4 = time.perf_counter()

        _ = loss.item()
        torch.npu.synchronize()
        t5 = time.perf_counter()

        times["total"].append(t5 - t0)
        times["data_h2d"].append(t1 - t0)
        times["forward"].append(t2 - t1)
        times["backward"].append(t3 - t2)
        times["step"].append(t4 - t3)
        times["item"].append(t5 - t4)

    print(f"\n=== {batches} batches benchmark (workers=0) ===")
    for k, v in times.items():
        arr = np.array(v) * 1000
        print(f"{k:12s}: mean={arr.mean():7.2f}ms  median={np.median(arr):7.2f}ms  max={arr.max():7.2f}ms")

    # 分析瓶颈
    means = {k: np.mean(v) * 1000 for k, v in times.items()}
    total = means["total"]
    print(f"\n=== 占比分析 ===")
    for k in ["data_h2d", "forward", "backward", "step", "item"]:
        print(f"{k:12s}: {means[k]:7.2f}ms ({means[k]/total*100:5.1f}%)")

    # 单独测量 CPU 端 __getitem__ 耗时
    t_getitem = []
    for i in range(100):
        t0 = time.perf_counter()
        _ = train_ds[i % len(train_ds)]
        t_getitem.append(time.perf_counter() - t0)
    arr = np.array(t_getitem) * 1000
    print(f"\n=== CPU __getitem__ 耗时 ===")
    print(f"{'getitem':12s}: mean={arr.mean():7.2f}ms  median={np.median(arr):7.2f}ms  max={arr.max():7.2f}ms")


if __name__ == "__main__":
    main()
