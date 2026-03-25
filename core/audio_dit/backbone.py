"""Audio DiT backbone for latent audio diffusion.

Training usage:
- audio_latents: [B, C, T'] (output of AudioVAEAdapter.encode)
- model(audio_latents, t, cond=None) -> predicted_noise [B, C, T']

Inference utilities:
- encode_waveform(waveform: [B,T]) -> latents [B,C,T']
- decode_waveform(latents: [B,C,T']) -> waveform_hat [B,T_dec] (approx)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from core.audio_dit.audio_vae_adapter import AudioVAEAdapter, AudioVAEAdapterConfig
from core.audio_dit.patch_embed import AudioPatchEmbed
from core.audio_dit.dit_block import AudioDiTBlock


@dataclass(frozen=True)
class AudioDiTConfig:
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    latent_channels: int = 16
    patch_size: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    audio_vae_downsample_factor: int = 8


class AudioDiTBackbone(nn.Module):
    def __init__(self, cfg: AudioDiTConfig) -> None:
        super().__init__()
        if cfg.hidden_size % cfg.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.cfg = cfg

        self.audio_vae = AudioVAEAdapter(
            AudioVAEAdapterConfig(
                latent_channels=cfg.latent_channels,
                downsample_factor=cfg.audio_vae_downsample_factor,
            )
        )

        self.patch_embed = AudioPatchEmbed(
            in_channels=cfg.latent_channels,
            hidden_size=cfg.hidden_size,
            patch_size=cfg.patch_size,
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )

        self.blocks = nn.ModuleList(
            [
                AudioDiTBlock(
                    dim=cfg.hidden_size,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm_out = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.final = nn.Linear(cfg.hidden_size, cfg.latent_channels * cfg.patch_size)

    def encode_waveform(self, waveform_bt: torch.Tensor) -> torch.Tensor:
        """
        waveform_bt: [B,T] float
        returns z_audio: [B,C,T']
        """
        if waveform_bt.ndim != 2:
            raise ValueError(f"waveform_bt must be [B,T], got {tuple(waveform_bt.shape)}")
        # Convert to dtype/device of audio_vae parameters.
        p = next(self.audio_vae.parameters())
        z = self.audio_vae.encode(waveform_bt.to(device=p.device, dtype=p.dtype))
        return z

    def decode_waveform(self, z_audio: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        p = next(self.audio_vae.parameters())
        wf = self.audio_vae.decode(z_audio.to(device=p.device, dtype=p.dtype), target_len=target_len)
        return wf

    @staticmethod
    def _timestep_embedding(timesteps: torch.Tensor, dim: int, dtype: torch.dtype) -> torch.Tensor:
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
        return emb.to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: [B, C, T'] audio latents
        """
        _ = cond
        if x.ndim != 3:
            raise ValueError(f"expected latents x [B,C,T'], got {tuple(x.shape)}")
        b, c, tt = x.shape
        if c != self.cfg.latent_channels:
            raise ValueError(f"expected in_channels={self.cfg.latent_channels}, got {c}")

        tokens, grid = self.patch_embed(x)
        t_emb = self._timestep_embedding(t, self.cfg.hidden_size, dtype=tokens.dtype)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)
        tokens = tokens + t_emb

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm_out(tokens)
        patches = self.final(tokens)  # [B,N,C*patch_size]
        out_latents = self.patch_embed.unpatchify(patches, grid, out_channels=self.cfg.latent_channels)
        if out_latents.shape[-1] != tt:
            out_latents = out_latents[..., :tt]
        return out_latents


def build_audio_dit_for_s1(
    hidden_size: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    latent_channels: int = 16,
    patch_size: int = 4,
    dropout: float = 0.0,
    mlp_ratio: float = 4.0,
    audio_vae_downsample_factor: int = 8,
) -> AudioDiTBackbone:
    return AudioDiTBackbone(
        AudioDiTConfig(
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            latent_channels=latent_channels,
            patch_size=patch_size,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            audio_vae_downsample_factor=audio_vae_downsample_factor,
        )
    )

