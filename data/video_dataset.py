"""MP4 video clips for training: [B, 3, T, H, W] in [-1, 1]."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    import decord

    decord.bridge.set_bridge("torch")
    _HAS_DECORD = True
except Exception:  # noqa: BLE001
    _HAS_DECORD = False

if not _HAS_DECORD:
    from torchvision.io import read_video


def _collect_video_paths(data_path: str | Path) -> list[Path]:
    p = Path(data_path).expanduser().resolve()
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if p.is_file():
        if p.suffix.lower() not in exts:
            raise ValueError(f"Unsupported video file: {p}")
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(data_path)
    out: list[Path] = []
    for ext in exts:
        out.extend(p.rglob(f"*{ext}"))
        out.extend(p.rglob(f"*{ext.upper()}"))
    return sorted({x.resolve() for x in out})


def _temporal_indices(num_frames: int, clip_len: int, rng: random.Random) -> list[int]:
    if num_frames <= 0:
        raise ValueError("video has no frames")
    if num_frames >= clip_len:
        start = rng.randint(0, num_frames - clip_len)
        return list(range(start, start + clip_len))
    idx = list(range(num_frames))
    out: list[int] = []
    while len(out) < clip_len:
        out.extend(idx)
    return out[:clip_len]


def _read_frames_decord(path: Path, indices: list[int]) -> Tensor:
    vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
    n = len(vr)
    if n == 0:
        raise RuntimeError(f"empty video: {path}")
    idx = [min(max(i, 0), n - 1) for i in indices]
    batch = vr.get_batch(idx).float() / 255.0
    if batch.ndim != 4:
        raise RuntimeError(f"unexpected decord shape: {tuple(batch.shape)}")
    return batch.permute(3, 0, 1, 2).contiguous()


class VideoMP4Dataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        clip_frames: int,
        height: int,
        width: int,
        augment_hflip: bool = True,
        color_jitter: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.paths = _collect_video_paths(data_path)
        if not self.paths:
            raise FileNotFoundError(f"No video files under {data_path}")
        self.clip_frames = clip_frames
        self.height = height
        self.width = width
        self.augment_hflip = augment_hflip
        self.color_jitter = color_jitter
        self._seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        rng = random.Random((self._seed * 100003 + index * 9187) & 0xFFFFFFFF)
        path = self.paths[index]
        if _HAS_DECORD:
            vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
            num_frames = len(vr)
        else:
            vid, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
            num_frames = int(vid.shape[0])

        idxs = _temporal_indices(num_frames, self.clip_frames, rng)
        if _HAS_DECORD:
            x = _read_frames_decord(path, idxs)
        else:
            n = num_frames
            pick = [min(max(i, 0), n - 1) for i in idxs]
            frames = vid[pick].float() / 255.0
            x = frames.permute(1, 0, 2, 3).contiguous()

        _, t, _, _ = x.shape
        x = F.interpolate(
            x.unsqueeze(0),
            size=(t, self.height, self.width),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        if self.augment_hflip and rng.random() < 0.5:
            x = torch.flip(x, dims=[-1])

        if self.color_jitter and rng.random() < 0.5:
            b = 0.9 + 0.2 * rng.random()
            c = 0.9 + 0.2 * rng.random()
            s = 0.9 + 0.2 * rng.random()
            x = x * b
            x = (x - x.mean(dim=(1, 2), keepdim=True)) * c + x.mean(dim=(1, 2), keepdim=True)
            gray = x.mean(dim=0, keepdim=True).expand_as(x)
            x = torch.lerp(gray, x, s)

        x = x.clamp(0.0, 1.0)
        x = x * 2.0 - 1.0
        return x


def build_dataloader(
    data_path: str | Path,
    batch_size: int,
    num_workers: int,
    frames: int,
    height: int,
    width: int,
    shuffle: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int | None = 2,
    persistent_workers: bool | None = None,
    augment_hflip: bool = True,
    color_jitter: bool = False,
    drop_last: bool = True,
    seed: int = 0,
) -> DataLoader:
    ds = VideoMP4Dataset(
        data_path=data_path,
        clip_frames=frames,
        height=height,
        width=width,
        augment_hflip=augment_hflip,
        color_jitter=color_jitter,
        seed=seed,
    )
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    kw: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kw["prefetch_factor"] = prefetch_factor
        kw["persistent_workers"] = persistent_workers
    return DataLoader(ds, **kw)


def infinite_dataloader(loader: DataLoader) -> Iterator[Tensor]:
    while True:
        for batch in loader:
            yield batch
