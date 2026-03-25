"""Audio patch embedding for latents [B,C,T'] -> tokens [B,N,D] via Conv1d."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AudioPatchGrid:
    t: int  # original latent length before padding
    n: int  # number of patches


class AudioPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, AudioPatchGrid]:
        """
        x: [B,C,T'] -> tokens [B,N,D]
        """
        if x.ndim != 3:
            raise ValueError(f"expected x [B,C,T'], got {tuple(x.shape)}")
        b, c, t = x.shape
        ps = self.patch_size
        if t % ps != 0:
            pad = ps - (t % ps)
            x = torch.nn.functional.pad(x, (0, pad))
        y = self.proj(x)  # [B,D,N]
        _, _, n = y.shape
        tokens = y.transpose(1, 2).contiguous()
        return tokens, AudioPatchGrid(t=t, n=n)

    def unpatchify(self, patches: torch.Tensor, grid: AudioPatchGrid, out_channels: int) -> torch.Tensor:
        """
        patches: [B,N, out_channels * patch_size] -> [B,out_channels,T']
        """
        if patches.ndim != 3:
            raise ValueError(f"expected patches [B,N,P], got {tuple(patches.shape)}")
        b, n, pdim = patches.shape
        if n != grid.n:
            raise ValueError("token count mismatch vs grid")
        ps = self.patch_size
        expected = out_channels * ps
        if pdim != expected:
            raise ValueError(f"patch dim {pdim} != out_channels*patch_size ({expected})")

        x = patches.view(b, n, out_channels, ps)  # [B,N,C,ps]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B,C,N,ps]
        x = x.view(b, out_channels, n * ps)  # padded length
        if x.shape[-1] != grid.t:
            x = x[..., : grid.t]
        return x

