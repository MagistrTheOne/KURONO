"""Temporal alignment helpers for audio-video token synchronization."""

import torch


def map_audio_time_to_video_time(audio_t: torch.Tensor, video_t_max: int) -> torch.Tensor:
    """Map audio timeline indices into video timeline index space."""
    if video_t_max <= 1:
        return torch.zeros_like(audio_t)
    audio_t = audio_t.float()
    scaled = audio_t / (audio_t.max().clamp(min=1.0))
    return (scaled * (video_t_max - 1)).long()


def temporal_align_indices(audio_indices: torch.Tensor, video_length: int) -> torch.Tensor:
    """Return aligned audio temporal indices for shared positional encoding."""
    return map_audio_time_to_video_time(audio_indices, video_t_max=video_length)

