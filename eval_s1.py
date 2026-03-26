"""Cheap S1 evaluation probes (no FVD/CLIP deps). Full metrics remain TODO."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from core.video_dit.wfvae_adapter import VAEAdapterConfig, WFVAEAdapter
from data.video_dataset import build_dataloader


def _dtype_from_precision(precision: str) -> torch.dtype:
    return torch.bfloat16 if precision == "bf16" else torch.float32


def temporal_l1(video: torch.Tensor) -> torch.Tensor:
    """Mean |x[:,:,t] - x[:,:,t-1]| over B,C,T>0,H,W."""
    if video.shape[2] < 2:
        return video.new_zeros(())
    d = (video[:, :, 1:] - video[:, :, :-1]).abs().mean()
    return d


def recon_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S1 probes: VAE recon + temporal stats on a data batch.")
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--frames", type=int, default=65)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--num-batches", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--mock-vae", action="store_true")
    p.add_argument("--wfvae-pretrained", type=str, default="", help="Local WF-VAE directory (prod weights only).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mock_vae and not args.wfvae_pretrained:
        raise ValueError("Provide --wfvae-pretrained or --mock-vae.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_precision(args.precision)

    vae = WFVAEAdapter(
        VAEAdapterConfig(
            from_pretrained=args.wfvae_pretrained,
            mock_mode=args.mock_vae,
            latent_channels=16,
        )
    )
    vae.build(device=device, dtype=dtype)

    dl = build_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=0,
        frames=args.frames,
        height=args.height,
        width=args.width,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=None,
        augment_hflip=False,
        color_jitter=False,
    )

    n = 0
    agg_recon = 0.0
    agg_t_in = 0.0
    agg_t_rec = 0.0

    it = iter(dl)
    while n < args.num_batches:
        try:
            batch = next(it)
        except StopIteration:
            break
        video = batch[0] if isinstance(batch, (list, tuple)) else batch
        video = video.to(device=device, dtype=dtype)
        with torch.no_grad():
            z = vae.encode(video)
            rec = vae.decode(z)
        agg_recon += float(recon_mse(video, rec).item())
        agg_t_in += float(temporal_l1(video).item())
        agg_t_rec += float(temporal_l1(rec).item())
        n += 1

    if n == 0:
        raise RuntimeError("No batches evaluated (empty dataloader?).")

    print(
        f"[eval_s1] batches={n} recon_mse={agg_recon / n:.6f} "
        f"temporal_l1_input={agg_t_in / n:.6f} temporal_l1_recon={agg_t_rec / n:.6f}"
    )
    print("[eval_s1] Note: FVD / CLIP-score not implemented here; add separate tooling for prod.")


if __name__ == "__main__":
    main()
