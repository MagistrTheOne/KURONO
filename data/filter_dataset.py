"""Score video clips, keep top percentile, write JSON manifest for training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.scoring import ClipScores, ScoreGates, ScoreWeights, combined_score
from data.video_dataset import _HAS_DECORD, _collect_video_paths


def _resize_score_space(x: torch.Tensor, score_height: int, score_width: int) -> torch.Tensor:
    _, t, _, _ = x.shape
    return F.interpolate(
        x.unsqueeze(0),
        size=(t, score_height, score_width),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)


def _score_chunk_task(payload: dict[str, Any]) -> list[dict[str, Any]]:
    path = payload["path"]
    starts: list[int] = payload["starts"]
    clip_frames: int = payload["clip_frames"]
    sh: int = payload["score_height"]
    sw: int = payload["score_width"]
    weights = ScoreWeights(
        motion=payload["w_motion"],
        sharpness=payload["w_sharpness"],
        entropy=payload["w_entropy"],
    )
    gates = ScoreGates(
        min_motion=payload["min_motion"],
        brightness_min=payload["brightness_min"],
        brightness_max=payload["brightness_max"],
    )
    out: list[dict[str, Any]] = []
    import decord

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    n = len(vr)
    if n == 0:
        return out
    for start in starts:
        if start + clip_frames > n:
            continue
        idx = list(range(start, start + clip_frames))
        idx = [min(max(i, 0), n - 1) for i in idx]
        batch = vr.get_batch(idx).float() / 255.0
        if batch.ndim != 4:
            continue
        x = batch.permute(3, 0, 1, 2).contiguous()
        x = _resize_score_space(x, sh, sw)
        res = combined_score(x, weights=weights, gates=gates)
        if res is None:
            continue
        rec = _scores_to_record(path, start, res)
        out.append(rec)
    return out


def _scores_to_record(path: str, start_frame: int, res: ClipScores) -> dict[str, Any]:
    return {
        "path": path,
        "start_frame": start_frame,
        "score": res.combined,
        "motion": res.motion,
        "sharpness": res.sharpness,
        "entropy": res.entropy,
        "brightness": res.brightness,
    }


def _iter_start_frames(num_frames: int, clip_frames: int, stride: int) -> list[int]:
    if num_frames < clip_frames:
        return []
    last = num_frames - clip_frames
    return list(range(0, last + 1, stride))


def _build_tasks_for_video(
    path: Path,
    *,
    clip_frames: int,
    clip_stride: int,
    clips_per_task: int,
    score_height: int,
    score_width: int,
    w_motion: float,
    w_sharpness: float,
    w_entropy: float,
    min_motion: float,
    brightness_min: float,
    brightness_max: float,
) -> tuple[list[dict[str, Any]], int]:
    import decord

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
    nf = len(vr)
    starts = _iter_start_frames(nf, clip_frames, clip_stride)
    if not starts:
        return [], 0
    path_str = str(path.resolve())
    tasks: list[dict[str, Any]] = []
    for i in range(0, len(starts), clips_per_task):
        chunk = starts[i : i + clips_per_task]
        tasks.append(
            {
                "path": path_str,
                "starts": chunk,
                "clip_frames": clip_frames,
                "score_height": score_height,
                "score_width": score_width,
                "w_motion": w_motion,
                "w_sharpness": w_sharpness,
                "w_entropy": w_entropy,
                "min_motion": min_motion,
                "brightness_min": brightness_min,
                "brightness_max": brightness_max,
            }
        )
    return tasks, len(starts)


def main() -> None:
    p = argparse.ArgumentParser(description="Filter videos by quality scores; write training manifest.")
    p.add_argument("--video-dir", type=str, required=True)
    p.add_argument("--out-json", type=str, required=True)
    p.add_argument("--clip-frames", type=int, required=True)
    p.add_argument("--clip-stride", type=int, default=-1, help="default: same as clip-frames")
    p.add_argument("--top-percentile", type=float, default=20.0, help="keep top K %% by combined score")
    p.add_argument("--score-height", type=int, default=256)
    p.add_argument("--score-width", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--clips-per-task", type=int, default=8, help="clips decoded per subprocess task (same video)")
    p.add_argument("--w-motion", type=float, default=1.0)
    p.add_argument("--w-sharpness", type=float, default=1.0)
    p.add_argument("--w-entropy", type=float, default=1.0)
    p.add_argument("--min-motion", type=float, default=0.0)
    p.add_argument("--brightness-min", type=float, default=0.05)
    p.add_argument("--brightness-max", type=float, default=0.95)
    a = p.parse_args()

    if not _HAS_DECORD:
        raise SystemExit("filter_dataset requires decord (sparse frame reads). Install decord and retry.")

    clip_stride = a.clip_frames if a.clip_stride < 0 else a.clip_stride
    video_dir = Path(a.video_dir).expanduser().resolve()
    paths = _collect_video_paths(video_dir)
    if not paths:
        raise SystemExit(f"No videos under {video_dir}")

    all_tasks: list[dict[str, Any]] = []
    total_candidates = 0
    for vp in paths:
        tasks, nc = _build_tasks_for_video(
            vp,
            clip_frames=a.clip_frames,
            clip_stride=clip_stride,
            clips_per_task=max(1, a.clips_per_task),
            score_height=a.score_height,
            score_width=a.score_width,
            w_motion=a.w_motion,
            w_sharpness=a.w_sharpness,
            w_entropy=a.w_entropy,
            min_motion=a.min_motion,
            brightness_min=a.brightness_min,
            brightness_max=a.brightness_max,
        )
        all_tasks.extend(tasks)
        total_candidates += nc

    records: list[dict[str, Any]] = []
    workers = max(1, a.num_workers)
    if workers == 1:
        for t in all_tasks:
            records.extend(_score_chunk_task(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_score_chunk_task, t) for t in all_tasks]
            for fut in as_completed(futs):
                records.extend(fut.result())

    records.sort(key=lambda r: r["score"], reverse=True)
    k = max(1, math.ceil(len(records) * (a.top_percentile / 100.0))) if records else 0
    kept = records[:k] if k else []

    unique_videos = sorted({r["path"] for r in kept})
    manifest = {
        "clips": kept,
        "filtered_video_paths": unique_videos,
        "config": {
            "video_dir": str(video_dir),
            "clip_frames": a.clip_frames,
            "clip_stride": clip_stride,
            "top_percentile": a.top_percentile,
            "score_height": a.score_height,
            "score_width": a.score_width,
            "weights": {"motion": a.w_motion, "sharpness": a.w_sharpness, "entropy": a.w_entropy},
            "gates": {
                "min_motion": a.min_motion,
                "brightness_min": a.brightness_min,
                "brightness_max": a.brightness_max,
            },
            "stats": {
                "num_videos": len(paths),
                "candidate_clip_slots": total_candidates,
                "after_gates": len(records),
                "kept": len(kept),
            },
        },
    }
    out_path = Path(a.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
