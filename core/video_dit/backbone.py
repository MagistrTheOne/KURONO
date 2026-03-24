"""Video DiT backbone: patchify -> transformer -> unpatchify.

Latents [B, C, T, H, W] in / out. Timestep + optional cond conditioning.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from core.video_dit.blocks import TransformerBlock
from core.video_dit.patch_embed import VideoPatchEmbed


class VideoDiTBackbone(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 12,
        depth: int = 12,
        patch_size: tuple[int, int, int] = (2, 4, 4),
        in_channels: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.patch_embed = VideoPatchEmbed(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.cond_patch = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cond_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_size)
        patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
        self.final = nn.Linear(hidden_size, in_channels * patch_vol)

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
        if x.ndim != 5:
            raise ValueError(f"expected x [B,C,T,H,W], got shape {tuple(x.shape)}")
        b, c, tt, th, tw = x.shape
        if c != self.in_channels:
            raise ValueError(f"expected in_channels={self.in_channels}, got {c}")
        pt, ph, pw = self.patch_size
        if tt % pt or th % ph or tw % pw:
            raise ValueError(
                f"latent spatial dims ({tt},{th},{tw}) must be divisible by patch_size {self.patch_size}"
            )

        tokens, grid = self.patch_embed(x)
        t_emb = self._timestep_embedding(t, self.hidden_size).to(device=tokens.device, dtype=tokens.dtype)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)
        tokens = tokens + t_emb

        if cond is not None:
            if cond.ndim == 2:
                tokens = tokens + self.cond_proj(cond).unsqueeze(1)
            elif cond.ndim == 5:
                if cond.shape[0] != b or cond.shape[2:] != (tt, th, tw):
                    raise ValueError(f"cond 5D must match x batch and spatial, got {tuple(cond.shape)} vs x")
                c_cond = self.cond_patch(cond)
                cond_tokens = c_cond.flatten(2).transpose(1, 2).contiguous()
                if cond_tokens.shape[1] != tokens.shape[1]:
                    raise ValueError("cond patch token count mismatch")
                tokens = tokens + cond_tokens.to(dtype=tokens.dtype)
            else:
                raise ValueError(f"cond must be None, 2D, or 5D, got ndim={cond.ndim}")

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm_out(tokens)
        patches = self.final(tokens)
        out = self.patch_embed.unpatchify(patches, grid)
        return out


def build_video_backbone_for_s1(
    hidden_size: int = 1024,
    num_heads: int = 12,
    depth: int = 12,
    patch_size: tuple[int, int, int] = (2, 4, 4),
    in_channels: int = 16,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
) -> VideoDiTBackbone:
    """Factory used by train_s1 / infer_s1. Extra kwargs optional; defaults are safe."""
    return VideoDiTBackbone(
        hidden_size=hidden_size,
        num_heads=num_heads,
        depth=depth,
        patch_size=patch_size,
        in_channels=in_channels,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )
