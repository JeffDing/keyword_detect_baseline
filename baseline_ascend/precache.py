"""预计算并缓存 mel 特征，加速 NPU 训练/推理。"""
from __future__ import annotations

import argparse
import os

from config import AUDIO, PATHS
from data import PairDataset, load_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=str, default="all",
                    help="train / dev_seen / dev_unseen / eval_seen / eval_unseen / all")
    ap.add_argument("--save-cache", action="store_true",
                    help="是否生成缓存文件（默认只统计）")
    ap.add_argument("--cache-dir", type=str, default=os.path.join(os.path.dirname(__file__), "cache"))
    args = ap.parse_args()

    subsets = []
    if args.subset == "all":
        subsets = [
            ("train", PATHS.train_zip, PATHS.train_csv, False),
            ("dev_seen", PATHS.dev_seen_zip, PATHS.dev_seen_csv, False),
            ("dev_unseen", PATHS.dev_unseen_zip, PATHS.dev_unseen_csv, False),
            ("eval_seen", PATHS.eval_seen_zip, PATHS.eval_seen_csv, True),
            ("eval_unseen", PATHS.eval_unseen_zip, PATHS.eval_unseen_csv, True),
        ]
    else:
        mapping = {
            "train": (PATHS.train_zip, PATHS.train_csv, False),
            "dev_seen": (PATHS.dev_seen_zip, PATHS.dev_seen_csv, False),
            "dev_unseen": (PATHS.dev_unseen_zip, PATHS.dev_unseen_csv, False),
            "eval_seen": (PATHS.eval_seen_zip, PATHS.eval_seen_csv, True),
            "eval_unseen": (PATHS.eval_unseen_zip, PATHS.eval_unseen_csv, True),
        }
        if args.subset not in mapping:
            raise ValueError(f"未知 subset: {args.subset}")
        subsets.append((args.subset,) + mapping[args.subset])

    for name, zip_path, csv_path, inference in subsets:
        print(f"[{name}] 加载样本列表...")
        pairs = load_pairs(csv_path, with_label=not inference)
        print(f"[{name}] 共 {len(pairs)} 对样本")

        if args.save_cache:
            ds = PairDataset(pairs, zip_path, AUDIO, inference=inference,
                             cache_dir=args.cache_dir, save_cache=True)
            count = 0
            for i in range(len(ds)):
                _ = ds[i]
                count += 1
                if count % 5000 == 0:
                    print(f"[{name}] 已缓存 {count}/{len(ds)}")
            print(f"[{name}] 缓存完成，共 {count} 对 -> {args.cache_dir}")
        else:
            print(f"[{name}] 预览前 3 对（不保存缓存）...")
            ds = PairDataset(pairs, zip_path, AUDIO, inference=inference)
            for i in range(min(3, len(ds))):
                e, q, label, pid = ds[i]
                print(f"  {pid}: e={tuple(e.shape)}, q={tuple(q.shape)}, label={label}")


if __name__ == "__main__":
    main()
