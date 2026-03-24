"""Video backbone interfaces for KURONO.

This file intentionally keeps a thin interface so we can swap:
- Latte-like dense DiT (S1)
- STDiT/SUV-style variants (S2+)
without rewriting the training loop.
"""

import math

import torch
from torch import nn


class VideoDiTBackbone(nn.Module):
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_in = nn.Conv3d(in_channels=16, out_channels=hidden_size, kernel_size=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.proj_out = nn.Conv3d(in_channels=hidden_size, out_channels=16, kernel_size=1)

    def _timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj_in(x)
        t_emb = self._timestep_embedding(t, self.hidden_size).to(device=x.device, dtype=x.dtype)
        t_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb
        if cond is not None:
            if cond.ndim == 2:
                cond_emb = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                cond_emb = cond
            x = x + cond_emb.to(device=x.device, dtype=x.dtype)
        # Placeholder: transformer stack is added in S1 implementation pass.
        x = self.proj_out(x)
        return x


def build_video_backbone_for_s1(hidden_size: int = 1024) -> VideoDiTBackbone:
    """Factory used by train_s1 to keep entrypoint stable."""
    return VideoDiTBackbone(hidden_size=hidden_size)

