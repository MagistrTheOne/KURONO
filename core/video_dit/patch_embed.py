"""3D patch embedding for video latents [B, C, T, H, W] -> tokens [B, N, D]."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PatchGrid:
    t: int
    h: int
    w: int


class VideoPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: tuple[int, int, int] = (2, 4, 4),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        pt, ph, pw = patch_size
        self.patch_size = (pt, ph, pw)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PatchGrid]:
        """
        x: [B, C, T, H, W]
        returns tokens [B, N, D], grid after patching
        """
        y = self.proj(x)
        b, d, gt, gh, gw = y.shape
        tokens = y.flatten(2).transpose(1, 2).contiguous()
        return tokens, PatchGrid(t=gt, h=gh, w=gw)

    def patch_dim(self) -> int:
        pt, ph, pw = self.patch_size
        return self.in_channels * pt * ph * pw

    def unpatchify(self, patches: torch.Tensor, grid: PatchGrid) -> torch.Tensor:
        """
        patches: [B, N, C * pt * ph * pw]
        returns [B, C, T, H, W]
        """
        b, n, pdim = patches.shape
        pt, ph, pw = self.patch_size
        c = self.in_channels
        expected = c * pt * ph * pw
        if pdim != expected:
            raise ValueError(f"patch dim {pdim} != in_channels * patch_vol = {expected}")
        if n != grid.t * grid.h * grid.w:
            raise ValueError(f"token count {n} != grid product {grid.t * grid.h * grid.w}")

        x = patches.view(b, grid.t, grid.h, grid.w, c, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(b, c, grid.t * pt, grid.h * ph, grid.w * pw)
        return x
