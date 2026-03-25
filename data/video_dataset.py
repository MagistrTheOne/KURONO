"""MP4 video clips for training: [B, 3, T, H, W] in [-1, 1]."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import decord

    decord.bridge.set_bridge("torch")
    _HAS_DECORD = True
except Exception:  # noqa: BLE001
    _HAS_DECORD = False

if not _HAS_DECORD:
    from torchvision.io import read_video


def _resample_waveform(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    if waveform.ndim != 1:
        raise ValueError(f"waveform must be [L], got {tuple(waveform.shape)}")
    l = waveform.shape[0]
    new_l = int(round(l * target_sr / orig_sr))
    if new_l <= 0:
        raise ValueError("invalid resampled length")
    x = waveform[None, None, :].to(dtype=torch.float32)
    y = F.interpolate(x, size=new_l, mode="linear", align_corners=False)
    return y.squeeze(0).squeeze(0)


def _decord_avg_fps(vr: "decord.VideoReader") -> float:
    # decord API varies slightly by version; use best-effort.
    if hasattr(vr, "get_avg_fps"):
        return float(vr.get_avg_fps())
    if hasattr(vr, "get_fps"):
        return float(vr.get_fps())
    # Fallback: estimate from first two frames.
    if len(vr) >= 2 and hasattr(vr, "get_frame_timestamp"):
        ts0 = vr.get_frame_timestamp(0)
        ts1 = vr.get_frame_timestamp(1)
        if isinstance(ts0, tuple):
            ts0 = ts0[0]
        if isinstance(ts1, tuple):
            ts1 = ts1[0]
        dt = float(ts1) - float(ts0)
        dt = max(dt, 1e-6)
        return 1.0 / dt
    return 30.0


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


def _finalize_video_clip(
    x: Tensor,
    *,
    height: int,
    width: int,
    rng: random.Random,
    augment_hflip: bool,
    color_jitter: bool,
) -> Tensor:
    _, t, _, _ = x.shape
    x = F.interpolate(
        x.unsqueeze(0),
        size=(t, height, width),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)

    if augment_hflip and rng.random() < 0.5:
        x = torch.flip(x, dims=[-1])

    if color_jitter and rng.random() < 0.5:
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


def _load_clip_from_video(path: Path, start_frame: int, clip_frames: int) -> Tensor:
    if _HAS_DECORD:
        vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
        n = len(vr)
        idx = list(range(start_frame, start_frame + clip_frames))
        idx = [min(max(i, 0), n - 1) for i in idx]
        batch = vr.get_batch(idx).float() / 255.0
        if batch.ndim != 4:
            raise RuntimeError(f"unexpected decord shape: {tuple(batch.shape)}")
        return batch.permute(3, 0, 1, 2).contiguous()
    vid, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
    n = int(vid.shape[0])
    pick = [min(max(i, 0), n - 1) for i in range(start_frame, start_frame + clip_frames)]
    frames = vid[pick].float() / 255.0
    return frames.permute(1, 0, 2, 3).contiguous()


class FilteredClipDataset(Dataset):
    """Training clips from `filter_dataset.py` JSON manifest (fixed start_frame per clip)."""

    def __init__(
        self,
        manifest_path: str | Path,
        clip_frames: int,
        height: int,
        width: int,
        augment_hflip: bool = True,
        color_jitter: bool = False,
        seed: int = 0,
        audio_enabled: bool = False,
        audio_sample_rate: int = 16000,
        assumed_video_fps: float = 30.0,
    ) -> None:
        super().__init__()
        mp = Path(manifest_path).expanduser().resolve()
        raw: dict[str, Any] = json.loads(mp.read_text(encoding="utf-8"))
        cfg = raw.get("config") or {}
        mf = cfg.get("clip_frames")
        if mf is not None and int(mf) != clip_frames:
            raise ValueError(
                f"manifest clip_frames={mf} does not match dataset clip_frames={clip_frames}"
            )
        self.entries: list[dict[str, Any]] = list(raw["clips"])
        if not self.entries:
            raise ValueError(f"No clips in manifest {manifest_path}")
        self.clip_frames = clip_frames
        self.height = height
        self.width = width
        self.augment_hflip = augment_hflip
        self.color_jitter = color_jitter
        self._seed = seed
        self.audio_enabled = audio_enabled
        self.audio_sample_rate = audio_sample_rate
        self.assumed_video_fps = assumed_video_fps
        self.audio_target_len = int(round((clip_frames / assumed_video_fps) * audio_sample_rate))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Any:
        rng = random.Random((self._seed * 100003 + index * 9187) & 0xFFFFFFFF)
        e = self.entries[index]
        path = Path(e["path"])
        start = int(e["start_frame"])
        if self.audio_enabled:
            if _HAS_DECORD:
                vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
                fps = _decord_avg_fps(vr)
            else:
                from torchvision.io import read_video

                _, _, info = read_video(str(path), start_pts=0, end_pts=1, pts_unit="sec", output_format="TCHW")
                fps = float(info.get("video_fps", self.assumed_video_fps))

            start_sec = start / fps
            end_sec = (start + self.clip_frames) / fps
            from torchvision.io import read_video

            vid, aud, info = read_video(
                str(path),
                start_pts=float(start_sec),
                end_pts=float(end_sec),
                pts_unit="sec",
                output_format="TCHW",
            )
            if aud.numel() == 0 or info.get("audio_fps", None) is None:
                raise ValueError(f"audio track missing in {path}")

            t_seg = int(vid.shape[0])
            if t_seg >= self.clip_frames:
                frames = vid[: self.clip_frames]
            else:
                pick = [min(i, t_seg - 1) for i in range(self.clip_frames)]
                frames = vid[pick]
            x = frames.float() / 255.0  # [T,3,H,W]
            x = x.permute(1, 0, 2, 3).contiguous()  # [3,T,H,W]
            video = _finalize_video_clip(
                x,
                height=self.height,
                width=self.width,
                rng=rng,
                augment_hflip=self.augment_hflip,
                color_jitter=self.color_jitter,
            )

            orig_sr = int(info["audio_fps"])
            wav = aud.float().mean(dim=0)  # [L]
            wav = _resample_waveform(wav, orig_sr=orig_sr, target_sr=self.audio_sample_rate)
            wav = wav.contiguous()
            if wav.shape[0] >= self.audio_target_len:
                wav = wav[: self.audio_target_len]
            else:
                wav = F.pad(wav, (0, self.audio_target_len - wav.shape[0]))
            mx = wav.abs().max().clamp(min=1e-6)
            if mx > 1.0:
                wav = wav / mx
            return video, wav

        x = _load_clip_from_video(path, start, self.clip_frames)
        return _finalize_video_clip(
            x,
            height=self.height,
            width=self.width,
            rng=rng,
            augment_hflip=self.augment_hflip,
            color_jitter=self.color_jitter,
        )

    @staticmethod
    def sampler_weights(entries: list[dict[str, Any]], *, power: float = 1.0, eps: float = 1e-8) -> Tensor:
        scores = [float(max(e.get("score", 1.0), eps)) ** float(power) for e in entries]
        return torch.tensor(scores, dtype=torch.double)


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
        audio_enabled: bool = False,
        audio_sample_rate: int = 16000,
        assumed_video_fps: float = 30.0,
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
        self.audio_enabled = audio_enabled
        self.audio_sample_rate = audio_sample_rate
        self.assumed_video_fps = assumed_video_fps
        self.audio_target_len = int(round((clip_frames / assumed_video_fps) * audio_sample_rate))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Any:
        rng = random.Random((self._seed * 100003 + index * 9187) & 0xFFFFFFFF)
        path = self.paths[index]
        if not self.audio_enabled:
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

            return _finalize_video_clip(
                x,
                height=self.height,
                width=self.width,
                rng=rng,
                augment_hflip=self.augment_hflip,
                color_jitter=self.color_jitter,
            )

        # Audio-enabled mode: extract aligned segment with torchvision.read_video.
        # This guarantees real audio loading; if audio is absent we hard-fail.
        from torchvision.io import read_video

        if _HAS_DECORD:
            vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
            num_frames = len(vr)
            fps = _decord_avg_fps(vr)
        else:
            vid_full, aud_full, info_full = read_video(
                str(path), start_pts=0, end_pts=None, pts_unit="sec", output_format="TCHW"
            )
            if aud_full.numel() == 0 or info_full.get("audio_fps", None) is None:
                raise ValueError(f"audio track missing in {path}")
            num_frames = int(vid_full.shape[0])
            fps = float(info_full.get("video_fps", self.assumed_video_fps))
            aud_full_mono = aud_full.float().mean(dim=0)
            audio_fps = int(info_full["audio_fps"])

        idxs = _temporal_indices(num_frames if num_frames > 0 else self.clip_frames, self.clip_frames, rng)
        start_frame = int(idxs[0])
        start_sec = start_frame / fps
        end_sec = (start_frame + self.clip_frames) / fps

        if _HAS_DECORD:
            vid, aud, info = read_video(
                str(path),
                start_pts=float(start_sec),
                end_pts=float(end_sec),
                pts_unit="sec",
                output_format="TCHW",
            )
            if aud.numel() == 0 or info.get("audio_fps", None) is None:
                raise ValueError(f"audio track missing in {path}")
            t_seg = int(vid.shape[0])
            if t_seg >= self.clip_frames:
                frames = vid[: self.clip_frames]
            else:
                pick = [min(i, t_seg - 1) for i in range(self.clip_frames)]
                frames = vid[pick]
            orig_sr = int(info["audio_fps"])
            aud_mono = aud.float().mean(dim=0)
        else:
            pick = [min(max(i, 0), num_frames - 1) for i in idxs]
            frames = vid_full[pick]
            aud_mono = aud_full_mono
            audio_fps = int(audio_fps)  # type: ignore[has-type]
            orig_sr = audio_fps

            # Slice audio by time; vid_full/audio_fps are from the same full decode.
            start_sample = int(round(start_sec * audio_fps))
            end_sample = int(round(end_sec * audio_fps))
            start_sample = max(start_sample, 0)
            end_sample = min(end_sample, aud_mono.shape[0])
            aud_mono = aud_mono[start_sample:end_sample]

        x = frames.float() / 255.0  # [T,3,H,W]
        x = x.permute(1, 0, 2, 3).contiguous()  # [3,T,H,W]
        video = _finalize_video_clip(
            x,
            height=self.height,
            width=self.width,
            rng=rng,
            augment_hflip=self.augment_hflip,
            color_jitter=self.color_jitter,
        )

        wav = _resample_waveform(aud_mono, orig_sr=orig_sr, target_sr=self.audio_sample_rate).contiguous()
        if wav.shape[0] >= self.audio_target_len:
            wav = wav[: self.audio_target_len]
        else:
            wav = F.pad(wav, (0, self.audio_target_len - wav.shape[0]))
        mx = wav.abs().max().clamp(min=1e-6)
        if mx > 1.0:
            wav = wav / mx
        return video, wav


def build_dataloader(
    data_path: str | Path | None,
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
    manifest_path: str | Path | None = None,
    use_weighted_sampling: bool = False,
    score_weight_power: float = 1.0,
    audio_enabled: bool = False,
    audio_sample_rate: int = 16000,
    assumed_video_fps: float = 30.0,
) -> DataLoader:
    if manifest_path is not None:
        ds: Dataset = FilteredClipDataset(
            manifest_path=manifest_path,
            clip_frames=frames,
            height=height,
            width=width,
            augment_hflip=augment_hflip,
            color_jitter=color_jitter,
            seed=seed,
            audio_enabled=audio_enabled,
            audio_sample_rate=audio_sample_rate,
            assumed_video_fps=assumed_video_fps,
        )
    else:
        if data_path is None:
            raise ValueError("data_path is required when manifest_path is None")
        ds = VideoMP4Dataset(
            data_path=data_path,
            clip_frames=frames,
            height=height,
            width=width,
            augment_hflip=augment_hflip,
            color_jitter=color_jitter,
            seed=seed,
            audio_enabled=audio_enabled,
            audio_sample_rate=audio_sample_rate,
            assumed_video_fps=assumed_video_fps,
        )
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    sampler: WeightedRandomSampler | None = None
    use_shuffle = shuffle
    if manifest_path is not None and use_weighted_sampling:
        if not isinstance(ds, FilteredClipDataset):
            raise TypeError("weighted sampling requires FilteredClipDataset")
        w = FilteredClipDataset.sampler_weights(ds.entries, power=score_weight_power)
        sampler = WeightedRandomSampler(w, num_samples=len(ds), replacement=True)
        use_shuffle = False
    kw: dict = {
        "batch_size": batch_size,
        "shuffle": use_shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if sampler is not None:
        kw["sampler"] = sampler
    if num_workers > 0:
        kw["prefetch_factor"] = prefetch_factor
        kw["persistent_workers"] = persistent_workers
    return DataLoader(ds, **kw)


def infinite_dataloader(loader: DataLoader) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch
