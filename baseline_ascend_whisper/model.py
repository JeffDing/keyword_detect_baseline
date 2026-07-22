"""Whisper 预训练编码器 + 逐帧匹配 + 对称 max-mean 软对齐。支持张量并行(TP)。"""
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


class ColumnParallelLinear(nn.Module):
    """列并行线性层：输出特征维度按列切分到 TP 组的各个 rank。

    当 tp_world_size=1 时，行为与 nn.Linear 完全相同。
    state_dict key 与 nn.Sequential 中对应位置的 nn.Linear 一致。
    """

    def __init__(self, in_features: int, out_features: int,
                 tp_rank: int = 0, tp_world_size: int = 1, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        assert out_features % tp_world_size == 0, \
            f"out_features({out_features}) must be divisible by tp_world_size({tp_world_size})"
        self.out_features_per_rank = out_features // tp_world_size

        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """行并行线性层：输入维度切分，输出 all-reduce 求和。

    当 tp_world_size=1 时，行为与 nn.Linear 完全相同。
    与 ColumnParallelLinear 配合使用：列并行输出 (B,*,out/tp) 直接送入行并行，
    行并行在输出维度上 all-reduce 恢复 (B,*,out)。
    """

    def __init__(self, in_features: int, out_features: int,
                 tp_rank: int = 0, tp_world_size: int = 1,
                 tp_group=None, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.tp_group = tp_group
        assert in_features % tp_world_size == 0, \
            f"in_features({in_features}) must be divisible by tp_world_size({tp_world_size})"
        self.in_features_per_rank = in_features // tp_world_size

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight)
        if self.tp_world_size > 1 and self.tp_group is not None:
            import torch.distributed as dist
            dist.all_reduce(out, group=self.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out


class FrameMatcher(nn.Module):
    """逐帧投影 + 对称 max-mean 软对齐 + 可学习温度。支持张量并行。

    TP 策略 (use_mlp=True):
        ColumnParallelLinear → GELU → RowParallelLinear
        列并行将输出维度切分，行并行 all-reduce 恢复完整维度，中间无需通信。

    TP 策略 (use_mlp=False):
        ColumnParallelLinear → all_gather → normalize
        单层列并行输出 (B,T,embed_dim/tp)，需 all_gather 恢复后归一化。
    """

    def __init__(self, whisper_dim: int, embed_dim: int = 256, use_mlp: bool = True,
                 tp_rank: int = 0, tp_world_size: int = 1, tp_group=None):
        super().__init__()
        self.use_mlp = use_mlp
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.tp_group = tp_group
        self.embed_dim = embed_dim

        if use_mlp:
            self.proj = nn.Sequential(
                ColumnParallelLinear(whisper_dim, embed_dim, tp_rank, tp_world_size),
                nn.GELU(),
                RowParallelLinear(embed_dim, embed_dim, tp_rank, tp_world_size, tp_group),
            )
        else:
            self.proj = ColumnParallelLinear(whisper_dim, embed_dim, tp_rank, tp_world_size)
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def _norm_feat(self, feat: torch.Tensor) -> torch.Tensor:
        proj_feat = self.proj(feat)
        if not self.use_mlp and self.tp_world_size > 1 and self.tp_group is not None:
            import torch.distributed as dist
            gathered = [torch.empty_like(proj_feat) for _ in range(self.tp_world_size)]
            dist.all_gather(gathered, proj_feat, group=self.tp_group)
            proj_feat = torch.cat(gathered, dim=-1)
        return F.normalize(proj_feat, dim=-1)

    def forward(self, e_feat: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        e = self._norm_feat(e_feat)
        q = self._norm_feat(q_feat)
        temp = self.log_temp.exp().clamp(min=0.1, max=100.0)
        sim = torch.matmul(e, q.transpose(1, 2)) / temp
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
    def __init__(self, whisper_model_name: str = "tiny", embed_dim: int = 256,
                 device: str = "cpu", use_mlp: bool = True,
                 tp_rank: int = 0, tp_world_size: int = 1, tp_group=None):
        super().__init__()
        self.backbone = WhisperBackbone(whisper_model_name, device=device)
        whisper_dim = self.backbone.n_audio_state
        self.matcher = FrameMatcher(whisper_dim, embed_dim, use_mlp=use_mlp,
                                    tp_rank=tp_rank, tp_world_size=tp_world_size,
                                    tp_group=tp_group)
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
