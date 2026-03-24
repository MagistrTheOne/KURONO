"""S1 structural training smoke run for KURONO.

No HF/model download by default. Intended to validate wiring on H200:
- video -> WF-VAE adapter encode
- latent -> VideoDiT backbone
- latent reconstruction loss
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from core.diffusion.noise_scheduler import NoiseScheduler
from core.training.ema import EMA
from core.video_dit.backbone import build_video_backbone_for_s1
from core.video_dit.wfvae_adapter import VAEAdapterConfig, WFVAEAdapter
from data.video_dataset import build_dataloader, infinite_dataloader


@dataclass
class S1Config:
    steps: int
    batch_size: int
    frames: int
    height: int
    width: int
    hidden_size: int
    lr: float
    precision: str
    device: str
    mock_vae: bool
    wfvae_repo: str
    wfvae_model_name: str
    wfvae_pretrained: str
    num_train_timesteps: int
    beta_start: float
    beta_end: float
    log_every: int
    save_sample_every: int
    sample_dir: str
    ema_decay: float
    checkpoint_every: int
    checkpoint_dir: str


def parse_args() -> S1Config:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--frames", type=int, default=65)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mock-vae", action="store_true", help="Use mock WF-VAE adapter")
    p.add_argument("--wfvae-repo", type=str, default="WF-VAE")
    p.add_argument("--wfvae-model-name", type=str, default="WFVAE")
    p.add_argument("--wfvae-pretrained", type=str, default="")
    p.add_argument("--num-train-timesteps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-sample-every", type=int, default=0)
    p.add_argument("--sample-dir", type=str, default="outputs/s1_samples")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    p.add_argument("--data-path", type=str, default="", help="Directory or video file; empty = random tensors")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--color-jitter", action="store_true")
    p.add_argument("--no-augment-hflip", action="store_true")
    a = p.parse_args()
    return S1Config(
        steps=a.steps,
        batch_size=a.batch_size,
        frames=a.frames,
        height=a.height,
        width=a.width,
        hidden_size=a.hidden_size,
        lr=a.lr,
        precision=a.precision,
        device=a.device,
        mock_vae=a.mock_vae or (a.wfvae_pretrained == ""),
        wfvae_repo=a.wfvae_repo,
        wfvae_model_name=a.wfvae_model_name,
        wfvae_pretrained=a.wfvae_pretrained,
        num_train_timesteps=a.num_train_timesteps,
        beta_start=a.beta_start,
        beta_end=a.beta_end,
        log_every=a.log_every,
        save_sample_every=a.save_sample_every,
        sample_dir=a.sample_dir,
        ema_decay=a.ema_decay,
        checkpoint_every=a.checkpoint_every,
        checkpoint_dir=a.checkpoint_dir,
        data_path=a.data_path,
        num_workers=a.num_workers,
        prefetch_factor=a.prefetch_factor,
        color_jitter=a.color_jitter,
        no_augment_hflip=a.no_augment_hflip,
    )


def _dtype_from_precision(precision: str) -> torch.dtype:
    return torch.bfloat16 if precision == "bf16" else torch.float32


def _random_video_batch(cfg: S1Config, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Structure-only random clips in normalized range expected by VAE.
    return torch.empty(
        (cfg.batch_size, 3, cfg.frames, cfg.height, cfg.width),
        device=device,
        dtype=dtype,
    ).uniform_(-1.0, 1.0)


def _expand_to_latent_rank(x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    while x.ndim < latents.ndim:
        x = x.unsqueeze(-1)
    return x


def _predict_x0_from_noise(
    scheduler: NoiseScheduler,
    z_t: torch.Tensor,
    pred_noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    alpha = scheduler.sqrt_alphas_cumprod[t].to(device=z_t.device, dtype=z_t.dtype)
    sigma = scheduler.sqrt_one_minus_alphas_cumprod[t].to(device=z_t.device, dtype=z_t.dtype)
    alpha = _expand_to_latent_rank(alpha, z_t)
    sigma = _expand_to_latent_rank(sigma, z_t)
    return (z_t - sigma * pred_noise) / alpha


def _save_checkpoint(path: str, model: nn.Module, optimizer: AdamW, ema: EMA, step: int, cfg: S1Config) -> None:
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "config": {
            "hidden_size": cfg.hidden_size,
            "num_train_timesteps": cfg.num_train_timesteps,
            "beta_start": cfg.beta_start,
            "beta_end": cfg.beta_end,
        },
    }
    torch.save(ckpt, path)


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_precision(cfg.precision)

    vae_cfg = VAEAdapterConfig(
        wfvae_repo_path=cfg.wfvae_repo,
        model_name=cfg.wfvae_model_name,
        from_pretrained=cfg.wfvae_pretrained,
        mock_mode=cfg.mock_vae,
        latent_channels=16,
    )
    vae = WFVAEAdapter(vae_cfg)
    vae.build(device=device, dtype=dtype)

    model: nn.Module = build_video_backbone_for_s1(hidden_size=cfg.hidden_size).to(device=device, dtype=dtype)
    scheduler = NoiseScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
    ).to(device=device)
    ema = EMA(model, decay=cfg.ema_decay)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    amp_enabled = (device.type == "cuda" and dtype == torch.bfloat16)
    print(
        f"[KURONO S1] device={device} precision={dtype} steps={cfg.steps} "
        f"shape=[{cfg.batch_size},3,{cfg.frames},{cfg.height},{cfg.width}] mock_vae={cfg.mock_vae} "
        f"data_path={cfg.data_path or '<random>'}"
    )

    data_iter = None
    if cfg.data_path:
        dl = build_dataloader(
            data_path=cfg.data_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            frames=cfg.frames,
            height=cfg.height,
            width=cfg.width,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=cfg.prefetch_factor,
            augment_hflip=not cfg.no_augment_hflip,
            color_jitter=cfg.color_jitter,
        )
        data_iter = infinite_dataloader(dl)

    pbar = tqdm(range(cfg.steps), desc="train_s1")
    pred = None
    for step in pbar:
        if data_iter is not None:
            video = next(data_iter).to(device=device, dtype=dtype, non_blocking=True)
        else:
            video = _random_video_batch(cfg, device=device, dtype=dtype)
        with torch.no_grad():
            latents = vae.encode(video)

        noise = torch.randn_like(latents)
        t = scheduler.sample_timesteps(batch_size=latents.shape[0], device=device)
        z_t = scheduler.add_noise(latents=latents, noise=noise, timesteps=t)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
            pred = model(z_t, t)
            loss = F.mse_loss(pred, noise)
        loss.backward()
        optimizer.step()
        ema.update(model)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            print(f"[KURONO S1] step={step} loss={loss.item():.6f}")

        if cfg.save_sample_every > 0 and (step % cfg.save_sample_every == 0 or step == cfg.steps - 1):
            os.makedirs(cfg.sample_dir, exist_ok=True)
            with torch.no_grad():
                ema.apply_shadow(model)
                pred_ema = model(z_t, t)
                pred_x0 = _predict_x0_from_noise(scheduler=scheduler, z_t=z_t, pred_noise=pred_ema, t=t)
                rec = vae.decode(pred_x0)
                ema.restore(model)
            sample_path = os.path.join(cfg.sample_dir, f"step_{step:06d}.pt")
            torch.save(rec[0].detach().float().cpu(), sample_path)

        if cfg.checkpoint_every > 0 and (step % cfg.checkpoint_every == 0 or step == cfg.steps - 1):
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"s1_step_{step:06d}.pt")
            _save_checkpoint(ckpt_path, model=model, optimizer=optimizer, ema=ema, step=step, cfg=cfg)

    # Structural decode check.
    with torch.no_grad():
        ema.apply_shadow(model)
        pred = model(z_t, t)
        final_latents = _predict_x0_from_noise(scheduler=scheduler, z_t=z_t, pred_noise=pred, t=t)
        rec = vae.decode(final_latents)
        ema.restore(model)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    final_ckpt_path = os.path.join(cfg.checkpoint_dir, "s1_last.pt")
    _save_checkpoint(final_ckpt_path, model=model, optimizer=optimizer, ema=ema, step=cfg.steps - 1, cfg=cfg)
    print(f"[KURONO S1] done. latent_shape={tuple(pred.shape)} rec_shape={tuple(rec.shape)}")


if __name__ == "__main__":
    main()

