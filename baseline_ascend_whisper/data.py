from __future__ import annotations

import csv
import io
import os
import zipfile
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from config import AudioConfig

_ZIP_CACHE: dict = {}


def _get_zip(path: str) -> zipfile.ZipFile:
    key = (os.getpid(), path)
    if key not in _ZIP_CACHE:
        _ZIP_CACHE[key] = zipfile.ZipFile(path, "r")
    return _ZIP_CACHE[key]


def read_wav(zip_path: str, name: str, sr: int) -> np.ndarray:
    data = _get_zip(zip_path).read(name)
    wav, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        t = torchaudio.functional.resample(
            torch.from_numpy(wav).unsqueeze(0), file_sr, sr)
        wav = t.squeeze(0).numpy()
    return wav.astype(np.float32)


def load_pairs(csv_path: str, with_label: bool) -> List[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            item = {"id": r["id"]}
            if with_label:
                item["label"] = int(r["label"])
            rows.append(item)
    return rows


def load_similar_pairs(csv_path: str) -> List[dict]:
    if not csv_path or not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            item = {"id": r["id"], "label": 0}
            if "similar_id" in r:
                item["similar_id"] = r["similar_id"]
            rows.append(item)
    return rows


class PairDataset(Dataset):
    def __init__(self, pairs: List[dict], zip_path: str, cfg: AudioConfig,
                 inference: bool = False):
        self.pairs = pairs
        self.zip_path = zip_path
        self.cfg = cfg
        self.inference = inference

    def __len__(self):
        return len(self.pairs)

    def _feat(self, wav_name: str) -> torch.Tensor:
        wav = read_wav(self.zip_path, wav_name, self.cfg.sample_rate)
        mel = whisper_log_mel(torch.from_numpy(wav), self.cfg.n_mels, self.cfg.max_frames)
        return mel

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        pid = p["id"]
        label = p.get("label", -1)
        e = self._feat(f"wav/{pid}_enroll.wav")
        q = self._feat(f"wav/{pid}_query.wav")
        return e, q, -1 if self.inference else label, pid


def collate(batch):
    es = torch.stack([b[0] for b in batch])
    qs = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    ids = [b[3] for b in batch]
    return es, qs, labels, ids


def whisper_log_mel(wav: torch.Tensor, n_mels: int, max_frames: int) -> torch.Tensor:
    import whisper
    mel = whisper.log_mel_spectrogram(wav, n_mels=n_mels)
    mel = whisper.pad_or_trim(mel, max_frames)
    return mel
