from __future__ import annotations

import csv
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from config import AudioConfig

_ZIP_CACHE: dict = {}
_ZIP_RAM_CACHE: dict = {}


def preload_zip(zip_path: str):
    if zip_path not in _ZIP_RAM_CACHE:
        print(f"[data] loading zip into RAM: {zip_path} ...")
        with open(zip_path, "rb") as f:
            _ZIP_RAM_CACHE[zip_path] = f.read()
        size_mb = len(_ZIP_RAM_CACHE[zip_path]) / 1e6
        print(f"[data] zip loaded: {size_mb:.0f} MB")


def _get_zip(path: str) -> zipfile.ZipFile:
    key = (os.getpid(), path)
    if key not in _ZIP_CACHE:
        if path not in _ZIP_RAM_CACHE:
            preload_zip(path)
        _ZIP_CACHE[key] = zipfile.ZipFile(io.BytesIO(_ZIP_RAM_CACHE[path]), "r")
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
        self.max_samples = cfg.sample_rate * 30

    def __len__(self):
        return len(self.pairs)

    def _read_wav(self, wav_name: str) -> torch.Tensor:
        wav = read_wav(self.zip_path, wav_name, self.cfg.sample_rate)
        max_samples = self.max_samples
        if len(wav) > max_samples:
            wav = wav[:max_samples]
        elif len(wav) < max_samples:
            wav = np.pad(wav, (0, max_samples - len(wav)))
        return torch.from_numpy(wav)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        pid = p["id"]
        label = p.get("label", -1)
        e = self._read_wav(f"wav/{pid}_enroll.wav")
        q = self._read_wav(f"wav/{pid}_query.wav")
        return e, q, -1 if self.inference else label, pid


def batch_read_pairs(pairs: List[dict], zip_path: str, cfg: AudioConfig,
                     inference: bool = False, max_workers: int = 8):
    max_samples = cfg.sample_rate * 30

    def read_one(item):
        pid = item["id"]
        label = item.get("label", -1)
        e_wav = read_wav(zip_path, f"wav/{pid}_enroll.wav", cfg.sample_rate)
        q_wav = read_wav(zip_path, f"wav/{pid}_query.wav", cfg.sample_rate)
        if len(e_wav) > max_samples:
            e_wav = e_wav[:max_samples]
        elif len(e_wav) < max_samples:
            e_wav = np.pad(e_wav, (0, max_samples - len(e_wav)))
        if len(q_wav) > max_samples:
            q_wav = q_wav[:max_samples]
        elif len(q_wav) < max_samples:
            q_wav = np.pad(q_wav, (0, max_samples - len(q_wav)))
        return e_wav, q_wav, -1 if inference else label, pid

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(read_one, pairs))

    es = torch.from_numpy(np.stack([r[0] for r in results]))
    qs = torch.from_numpy(np.stack([r[1] for r in results]))
    labels = torch.tensor([r[2] for r in results], dtype=torch.float32)
    ids = [r[3] for r in results]
    return es, qs, labels, ids


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


def batch_mel_spectrogram(wavs: torch.Tensor, n_mels: int = 80,
                          max_frames: int = 3000) -> torch.Tensor:
    import math
    import whisper
    from whisper.audio import mel_filters
    mel_fb = mel_filters(wavs.device, n_mels)
    window = torch.hann_window(400, device=wavs.device, dtype=wavs.dtype)
    spec = torch.stft(
        wavs, n_fft=400, hop_length=160, win_length=400,
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
    return log_spec
