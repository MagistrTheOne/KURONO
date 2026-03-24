"""DiT block: residual Attention(LN) + residual GeGLU MoE FFN(LN)."""

from __future__ import annotations

import torch
from torch import nn

from core.moe.moe_block import MoEFFN
from core.video_dit.blocks import VideoDiTAttention


class VideoDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        moe_capacity_factor: float | None = 1.25,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = VideoDiTAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.moe = MoEFFN(
            dim,
            num_experts=num_experts,
            mlp_ratio=mlp_ratio,
            top_k=2,
            capacity_factor=moe_capacity_factor,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        x = x + self.attn(h, cos, sin)
        h2 = self.norm2(x)
        y, aux = self.moe(h2)
        return x + y, aux
