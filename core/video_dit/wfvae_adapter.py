"""WF-VAE adapter for KURONO (internal core.vae, no external repo on sys.path)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class VAEAdapterConfig:
    wfvae_repo_path: str = "WF-VAE"
    model_name: str = "WFVAE"
    from_pretrained: str = ""
    use_tiling: bool = False
    mock_mode: bool = False
    latent_channels: int = 16


class WFVAEAdapter(nn.Module):
    def __init__(self, cfg: VAEAdapterConfig):
        super().__init__()
        self.cfg = cfg
        self._vae = None
        self._mock_in_proj = nn.Conv3d(3, cfg.latent_channels, kernel_size=1)
        self._mock_out_proj = nn.Conv3d(cfg.latent_channels, 3, kernel_size=1)

    def build(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.cfg.mock_mode:
            self.to(device=device, dtype=dtype)
            return

        import core.vae.modeling_wfvae  # noqa: F401 — register "WFVAE"

        from core.vae.model import ModelRegistry

        model_cls = ModelRegistry.get_model(self.cfg.model_name)
        if model_cls is None:
            raise ValueError(f"Unknown WF-VAE model name: {self.cfg.model_name}")
        if not self.cfg.from_pretrained:
            raise ValueError("from_pretrained must be set when mock_mode is False")

        pretrained = Path(self.cfg.from_pretrained).expanduser()
        if not pretrained.exists():
            raise FileNotFoundError(f"Pretrained path not found: {pretrained}")

        vae = model_cls.from_pretrained(str(pretrained))
        if self.cfg.use_tiling and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        self._vae = vae.to(device=device, dtype=dtype).eval()

    @torch.no_grad()
    def encode(self, video_bcthw: torch.Tensor) -> torch.Tensor:
        """video_bcthw: [B, 3, T, H, W] -> latent [B, C, T/4, H/8, W/8]."""
        if self.cfg.mock_mode:
            x = self._mock_in_proj(video_bcthw)
            x = F.avg_pool3d(x, kernel_size=(4, 8, 8), stride=(4, 8, 8))
            return x

        if self._vae is None:
            raise RuntimeError("WFVAEAdapter.build() must be called before encode()")
        return self._vae.encode(video_bcthw).latent_dist.sample()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latent -> reconstructed video [B, 3, T, H, W]."""
        if self.cfg.mock_mode:
            x = F.interpolate(latents, scale_factor=(4, 8, 8), mode="trilinear", align_corners=False)
            x = self._mock_out_proj(x)
            return torch.clamp(x, -1.0, 1.0)

        if self._vae is None:
            raise RuntimeError("WFVAEAdapter.build() must be called before decode()")
        return self._vae.decode(latents).sample
