"""Video clip quality scores on CPU. Tensors are [C, T, H, W] float32 in [0, 1]."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ScoreWeights:
    motion: float = 1.0
    sharpness: float = 1.0
    entropy: float = 1.0


@dataclass(frozen=True)
class ScoreGates:
    min_motion: float = 0.0
    brightness_min: float = 0.05
    brightness_max: float = 0.95


class ClipScores(NamedTuple):
    combined: float
    motion: float
    sharpness: float
    entropy: float
    brightness: float


# 3x3 Laplacian (single channel)
_LAPLACIAN_KERNEL = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


def _ensure_float01(video: Tensor) -> Tensor:
    v = video.detach().float().contiguous()
    if v.ndim != 4:
        raise ValueError(f"expected [C,T,H,W], got {tuple(v.shape)}")
    return v


def _luminance(video: Tensor) -> Tensor:
    """[C,T,H,W] -> [T,H,W]"""
    v = _ensure_float01(video)
    if v.shape[0] == 3:
        r, g, b = v[0], v[1], v[2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    if v.shape[0] == 1:
        return v[0]
    gray = v.mean(dim=0)
    return gray


def motion_score(video: Tensor) -> Tensor:
    y = _luminance(video)
    if y.shape[0] < 2:
        return torch.zeros((), dtype=y.dtype, device=y.device)
    d = y[1:] - y[:-1]
    return d.abs().mean()


def sharpness_score(video: Tensor) -> Tensor:
    y = _luminance(video).unsqueeze(1)  # [T,1,H,W]
    t, _, h, w = y.shape
    k = _LAPLACIAN_KERNEL.to(device=y.device, dtype=y.dtype)
    resp = F.conv2d(y.reshape(1, t, h, w), k.expand(t, 1, 3, 3), groups=t, padding=1)
    resp = resp.view(t, h, w)
    return resp.pow(2).mean(dim=(1, 2)).mean()


def brightness_score(video: Tensor) -> Tensor:
    return _luminance(video).mean()


def _entropy_hist_256(gray_flat: Tensor) -> Tensor:
    hist = torch.histc(gray_flat, bins=256, min=0.0, max=1.0)
    p = hist.float()
    tot = p.sum().clamp(min=1.0)
    p = p / tot
    mask = p > 0
    return -(p[mask] * p[mask].log()).sum()


def entropy_score(video: Tensor) -> Tensor:
    y = _luminance(video)
    entropies = []
    for ti in range(y.shape[0]):
        entropies.append(_entropy_hist_256(y[ti].reshape(-1).clamp(0.0, 1.0)))
    return torch.stack(entropies).mean()


def combined_score(
    video: Tensor,
    *,
    weights: ScoreWeights = ScoreWeights(),
    gates: ScoreGates = ScoreGates(),
) -> ClipScores | None:
    m = motion_score(video)
    s = sharpness_score(video)
    e = entropy_score(video)
    b = brightness_score(video)
    if (m < gates.min_motion) or (b < gates.brightness_min) or (b > gates.brightness_max):
        return None
    c = weights.motion * m + weights.sharpness * s + weights.entropy * e
    return ClipScores(
        combined=float(c.item()),
        motion=float(m.item()),
        sharpness=float(s.item()),
        entropy=float(e.item()),
        brightness=float(b.item()),
    )


def batch_combined_scores(
    videos: Tensor,
    *,
    weights: ScoreWeights = ScoreWeights(),
    gates: ScoreGates = ScoreGates(),
) -> list[ClipScores | None]:
    """videos: [N, C, T, H, W] in [0,1]."""
    if videos.ndim != 5:
        raise ValueError(f"expected [N,C,T,H,W], got {tuple(videos.shape)}")
    out: list[ClipScores | None] = []
    for i in range(videos.shape[0]):
        out.append(combined_score(videos[i], weights=weights, gates=gates))
    return out
