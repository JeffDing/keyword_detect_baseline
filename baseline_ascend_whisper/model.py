"""Whisper 预训练编码器 + 逐帧匹配 + 对称 max-mean 软对齐。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperBackbone(nn.Module):
    """Whisper 预训练音频编码器，输出逐帧 embedding。"""

    def __init__(self, model_name: str = "tiny", device: str = "cpu"):
        super().__init__()
        import whisper

        self.whisper = whisper.load_model(model_name, device="cpu")
        self.whisper.eval()
        self.n_mels = self.whisper.dims.n_mels
        self.n_audio_ctx = self.whisper.dims.n_audio_ctx
        self.n_audio_state = self.whisper.dims.n_audio_state

        for param in self.whisper.encoder.parameters():
            param.requires_grad = False

        encoder_param = next(self.whisper.encoder.parameters())
        print(f"[WhisperBackbone] target_device={device}, encoder_param_device_before_to={encoder_param.device}")

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        feat = self.whisper.encoder(mel.detach())
        return feat


class FrameMatcher(nn.Module):
    """逐帧投影 + 对称 max-mean 软对齐。"""

    def __init__(self, whisper_dim: int, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(whisper_dim, embed_dim)

    def _norm_feat(self, feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(feat), dim=-1)

    def forward(self, e_feat: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        e = self._norm_feat(e_feat)
        q = self._norm_feat(q_feat)
        sim = torch.matmul(e, q.transpose(1, 2))
        score_e2q = sim.max(dim=2)[0].mean(dim=1)
        score_q2e = sim.max(dim=1)[0].mean(dim=1)
        return (score_e2q + score_q2e) / 2

    @torch.no_grad()
    def global_similarity(self, e_feat: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        e = self._norm_feat(e_feat)
        q = self._norm_feat(q_feat)
        e_g = e.mean(dim=1)
        q_g = q.mean(dim=1)
        return torch.matmul(e_g, q_g.t())


class SiameseKWS(nn.Module):
    def __init__(self, whisper_model_name: str = "tiny", embed_dim: int = 64, device: str = "cpu"):
        super().__init__()
        self.backbone = WhisperBackbone(whisper_model_name, device=device)
        whisper_dim = self.backbone.n_audio_state
        self.matcher = FrameMatcher(whisper_dim, embed_dim)
        self.scale = nn.Parameter(torch.tensor(8.0, device=device))
        self.bias = nn.Parameter(torch.tensor(0.0, device=device))
        self.to(device)

    def forward(self, enroll: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        e_feat = self.backbone(enroll)
        q_feat = self.backbone(query)
        score = self.matcher(e_feat, q_feat)
        return self.scale * score + self.bias

    @torch.no_grad()
    def get_embeddings(self, enroll: torch.Tensor, query: torch.Tensor):
        e_feat = self.backbone(enroll)
        q_feat = self.backbone(query)
        return e_feat, q_feat
