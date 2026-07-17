from __future__ import annotations

import os
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80           
    max_frames: int = 3000     
    whisper_model_name: str = "tiny"


@dataclass
class TrainConfig:
    embed_dim: int = 64
    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 10
    lr: float = 1e-3
    pos_weight: float = 4.0   
    train_subset: int = 500000  
    seed: int = 42
    log_every: int = 100
    hard_mining_every: int = 5
    hard_top_k: int = 5
    hard_weight: float = 2.0
    hard_use_global: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2


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
    similar_word_csv: str = field(default="")

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
        self.similar_word_csv = os.path.join(r, "similar_word_pairs.csv")


PATHS = Paths()
AUDIO = AudioConfig()
TRAIN = TrainConfig()
