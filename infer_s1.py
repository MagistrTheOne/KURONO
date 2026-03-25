from __future__ import annotations

import argparse
import os

import torch

from core.diffusion.noise_scheduler import NoiseScheduler
from core.diffusion.sampler import DDPMSampler, DDIMSampler
from core.training.ema import EMA
from core.video_dit.backbone import build_video_backbone_for_s1
from core.video_dit.wfvae_adapter import VAEAdapterConfig, WFVAEAdapter


def _dtype_from_precision(precision: str) -> torch.dtype:
    return torch.bfloat16 if precision == "bf16" else torch.float32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--dit-depth", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--moe-capacity-factor", type=float, default=1.25)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--out", type=str, default="outputs/infer_s1_video.pt")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--mock-vae", action="store_true")
    p.add_argument("--wfvae-repo", type=str, default="WF-VAE")
    p.add_argument("--wfvae-model-name", type=str, default="WFVAE")
    p.add_argument("--wfvae-pretrained", type=str, default="")
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mock_vae and not args.wfvae_pretrained:
        raise ValueError("WF-VAE pretrained path is required unless --mock-vae is set.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_precision(args.precision)

    vae_cfg = VAEAdapterConfig(
        wfvae_repo_path=args.wfvae_repo,
        model_name=args.wfvae_model_name,
        from_pretrained=args.wfvae_pretrained,
        mock_mode=args.mock_vae,
        latent_channels=16,
    )
    vae = WFVAEAdapter(vae_cfg)
    vae.build(device=device, dtype=dtype)

    ckpt = None
    c: dict = {}
    if args.checkpoint:
        try:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location=device)
        c = ckpt.get("config") or {}

    model = build_video_backbone_for_s1(
        hidden_size=int(c.get("hidden_size", args.hidden_size)),
        num_heads=int(c.get("num_heads", args.num_heads)),
        depth=int(c.get("dit_depth", args.dit_depth)),
        num_experts=int(c.get("num_experts", args.num_experts)),
        moe_capacity_factor=args.moe_capacity_factor,
        dropout=args.dropout,
    ).to(device=device, dtype=dtype).eval()
    ema = EMA(model)

    scheduler = NoiseScheduler(
        num_train_timesteps=args.steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    ).to(device=device)
    if args.sampler == "ddim":
        sampler = DDIMSampler(scheduler=scheduler, num_inference_steps=args.num_inference_steps, eta=0.0)
    else:
        sampler = DDPMSampler(scheduler=scheduler)

    use_ema = args.use_ema or not args.no_ema

    ema_applied = False
    if ckpt is not None:
        model.load_state_dict(ckpt["model"], strict=True)
        if use_ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
            ema.apply_shadow(model)
            ema_applied = True

    latent_shape = (
        args.batch_size,
        16,
        args.frames // 4,
        args.height // 8,
        args.width // 8,
    )
    cond = None
    try:
        with torch.no_grad():
            sampled_latents = sampler.sample(
                model=model,
                shape=latent_shape,
                device=device,
                dtype=dtype,
                cond=cond,
                guidance_scale=args.guidance_scale,
            )
            video = vae.decode(sampled_latents)
    finally:
        if ema_applied:
            ema.restore(model)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(video.detach().float().cpu(), args.out)


if __name__ == "__main__":
    main()
