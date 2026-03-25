"""Audio DiT blocks: attention only."""

from __future__ import annotations

import math

import torch
from torch import nn


class AudioSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,N,D]
        """
        if x.ndim != 3:
            raise ValueError(f"expected x [B,N,D], got {tuple(x.shape)}")
        b, n, d = x.shape
        qkv = self.qkv(x)  # [B,N,3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,hd]
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )  # [B,H,N,hd]
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class AudioFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

