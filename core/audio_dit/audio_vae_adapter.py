"""Minimal real Audio VAE adapter (Conv1D encoder/decoder).

API:
- encode(waveform: [B,T]) -> z_audio: [B,C,T']
- decode(z_audio: [B,C,T']) -> waveform_hat: [B,T_dec]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AudioVAEAdapterConfig:
    latent_channels: int = 16
    downsample_factor: int = 8
    # Ensure the input length is padded to a multiple of (downsample_factor).
    pad_to_multiple: bool = True


class AudioVAEAdapter(nn.Module):
    def __init__(self, cfg: AudioVAEAdapterConfig) -> None:
        super().__init__()
        if cfg.downsample_factor not in (2, 4, 8, 16):
            raise ValueError("downsample_factor must be one of {2,4,8,16}")

        self.cfg = cfg
        ds = cfg.downsample_factor
        # Build a simple strided encoder/decoder with exact length doubling.
        # Each stage: kernel=4, stride=2, padding=1 => out_len = 2*in_len for decoder.
        stages = int(torch.log2(torch.tensor(ds)).item())
        if stages < 1:
            raise ValueError("invalid downsample_factor")

        enc = []
        in_ch = 1
        for _ in range(stages):
            enc.append(nn.Conv1d(in_ch, cfg.latent_channels, kernel_size=4, stride=2, padding=1))
            enc.append(nn.GELU())
            in_ch = cfg.latent_channels
        self.encoder = nn.Sequential(*enc)

        dec = []
        in_ch = cfg.latent_channels
        for i in range(stages - 1):
            dec.append(
                nn.ConvTranspose1d(in_ch, cfg.latent_channels, kernel_size=4, stride=2, padding=1)
            )
            dec.append(nn.GELU())
            in_ch = cfg.latent_channels
        # Final stage to mono waveform.
        dec.append(nn.ConvTranspose1d(cfg.latent_channels, 1, kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*dec)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
        t = x.shape[-1]
        rem = t % multiple
        if rem == 0:
            return x, 0
        pad = multiple - rem
        x = F.pad(x, (0, pad))
        return x, pad

    @torch.no_grad()
    def encode(self, waveform_bt: torch.Tensor) -> torch.Tensor:
        if waveform_bt.ndim != 2:
            raise ValueError(f"waveform must be [B,T], got {tuple(waveform_bt.shape)}")
        x = waveform_bt
        if self.cfg.pad_to_multiple and self.cfg.downsample_factor > 1:
            x, _ = self._pad_to_multiple(x, self.cfg.downsample_factor)
        x = x.unsqueeze(1)  # [B,1,T]
        z = self.encoder(x)
        return z

    @torch.no_grad()
    def decode(self, z_audio: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        if z_audio.ndim != 3:
            raise ValueError(f"z_audio must be [B,C,T'], got {tuple(z_audio.shape)}")
        x = self.decoder(z_audio)  # [B,1,T_dec]
        x = x.squeeze(1)
        if target_len is not None:
            if x.shape[-1] >= target_len:
                x = x[..., :target_len]
            else:
                x = F.pad(x, (0, target_len - x.shape[-1]))
        return x

