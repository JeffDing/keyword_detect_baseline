from __future__ import annotations

import os
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 40          
    max_frames: int = 100      


@dataclass
class TrainConfig:
    embed_dim: int = 64
    batch_size: int = 128
    num_workers: int = 8
    epochs: int = 10
    lr: float = 1e-3
    pos_weight: float = 4.0   # 训练集正负比约 1:4
    train_subset: int = 500000  
    seed: int = 42
    log_every: int = 100


@dataclass
class Paths:
    root: str = ROOT
    train_zip: str = field(default="")
    train_csv: str = field(default="")
    dev_seen_zip: str = field(default="")
    dev_seen_csv: str = field(default="")
    dev_unseen_zip: str = field(default="")
    dev_unseen_csv: str = field(default="")
    eval_seen_zip: str = field(default="")
    eval_seen_csv: str = field(default="")
    eval_unseen_zip: str = field(default="")
    eval_unseen_csv: str = field(default="")
    ckpt_dir: str = field(default="")

    def __post_init__(self):
        r = self.root
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        self.dev_seen_zip = os.path.join(r, "dev", "dev_seen", "wav.zip")
        self.dev_seen_csv = os.path.join(r, "dev", "dev_seen", "dev_seen_label.csv")
        self.dev_unseen_zip = os.path.join(r, "dev", "dev_unseen", "wav.zip")
        self.dev_unseen_csv = os.path.join(r, "dev", "dev_unseen", "dev_unseen_label.csv")
        self.eval_seen_zip = os.path.join(r, "eval", "eval_seen", "wav.zip")
        self.eval_seen_csv = os.path.join(r, "evalcsv_without_label", "eval_seen_without_label.csv")
        self.eval_unseen_zip = os.path.join(r, "eval", "eval_unseen", "wav.zip")
        self.eval_unseen_csv = os.path.join(r, "evalcsv_without_label", "eval_unseen_without_label.csv")
        self.ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")


PATHS = Paths()
AUDIO = AudioConfig()
TRAIN = TrainConfig()
