"""Audio DiT placeholder backbone for staged integration."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class AudioBatch:
    latents: torch.Tensor  # [B, C, T]
    text_emb: Optional[torch.Tensor] = None


class AudioDiTBackbone(nn.Module):
    def __init__(self, hidden_size: int = 768, in_channels: int = 64):
        super().__init__()
        self.proj_in = nn.Conv1d(in_channels=in_channels, out_channels=hidden_size, kernel_size=1)
        self.proj_out = nn.Conv1d(in_channels=hidden_size, out_channels=in_channels, kernel_size=1)

    def forward(self, batch: AudioBatch) -> torch.Tensor:
        x = self.proj_in(batch.latents)
        # Placeholder: add transformer stack in S3.
        x = self.proj_out(x)
        return x

