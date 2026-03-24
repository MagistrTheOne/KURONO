"""Spatio-temporal attention with 3D RoPE; tokens [B, N, D] ordered (t, h, w) with W fastest."""

from __future__ import annotations

import math

import torch
from torch import nn

from core.video_dit.patch_embed import PatchGrid


def _rope_split_dims(head_dim: int) -> tuple[int, int, int]:
    assert head_dim % 2 == 0 and head_dim >= 6
    a = 2 * max(1, (head_dim // 3) // 2)
    b = 2 * max(1, ((head_dim - a) // 2) // 2)
    c = head_dim - a - b
    if c < 2:
        shift = 2 - c
        b = max(2, b - (shift - (shift % 2)))
        c = head_dim - a - b
    while c % 2:
        b -= 2
        c += 2
    assert a + b + c == head_dim and a % 2 == 0 and b % 2 == 0 and c % 2 == 0
    return a, b, c


def _axis_freqs(pos: torch.Tensor, dim: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    half = dim // 2
    idx = torch.arange(half, device=device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (idx / max(half, 1)))
    ang = pos.float().unsqueeze(-1) * freqs
    return ang.cos().to(pos.dtype), ang.sin().to(pos.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos_f = torch.cat([cos, cos], dim=-1).unsqueeze(1)
    sin_f = torch.cat([sin, sin], dim=-1).unsqueeze(1)
    q_embed = (q * cos_f) + (rotate_half(q) * sin_f)
    k_embed = (k * cos_f) + (rotate_half(k) * sin_f)
    return q_embed, k_embed


def build_video_rope_cos_sin(
    grid: PatchGrid,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt, dh, dw = _rope_split_dims(head_dim)
    n = grid.t * grid.h * grid.w
    ar = torch.arange(n, device=device)
    gw = grid.w
    gh = grid.h
    w_idx = ar % gw
    h_idx = (ar // gw) % gh
    t_idx = ar // (gw * gh)

    ct, st = _axis_freqs(t_idx, dt, theta, device)
    ch, sh = _axis_freqs(h_idx, dh, theta, device)
    cw, sw = _axis_freqs(w_idx, dw, theta, device)
    cos = torch.cat([ct, ch, cw], dim=-1)
    sin = torch.cat([st, sh, sw], dim=-1)
    return cos.to(dtype).unsqueeze(0), sin.to(dtype).unsqueeze(0)


class VideoDiTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.drop(self.proj(out))
