from __future__ import annotations

import torch
from torch import nn

from core.diffusion.noise_scheduler import NoiseScheduler


def _predict_noise_with_cfg(
    model: nn.Module,
    z_t: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor | None,
    guidance_scale: float,
) -> torch.Tensor:
    if cond is None:
        return model(z_t, t, cond=None)

    pred_uncond = model(z_t, t, cond=None)
    pred_cond = model(z_t, t, cond=cond)
    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


class DDPMSampler:
    def __init__(self, scheduler: NoiseScheduler) -> None:
        self.scheduler = scheduler

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
        cond: torch.Tensor | None = None,
        uncond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        _ = uncond
        z_t = torch.randn(shape, device=device, dtype=dtype)
        num_steps = self.scheduler.num_train_timesteps

        for step in reversed(range(num_steps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            pred_noise = _predict_noise_with_cfg(
                model=model,
                z_t=z_t,
                t=t,
                cond=cond,
                guidance_scale=guidance_scale,
            )

            alpha_t = self.scheduler.alphas[step].to(device=device, dtype=dtype)
            alpha_cumprod_t = self.scheduler.alpha_cumprod[step].to(device=device, dtype=dtype)
            beta_t = self.scheduler.betas[step].to(device=device, dtype=dtype)

            one_over_sqrt_alpha_t = torch.rsqrt(alpha_t)
            coeff = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)
            mean = one_over_sqrt_alpha_t * (z_t - coeff * pred_noise)

            if step > 0:
                noise = torch.randn_like(z_t)
                z_t = mean + torch.sqrt(beta_t) * noise
            else:
                z_t = mean

        return z_t


class DDIMSampler:
    def __init__(self, scheduler: NoiseScheduler, num_inference_steps: int = 50, eta: float = 0.0) -> None:
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta

    def _inference_timesteps(self) -> list[int]:
        train_steps = self.scheduler.num_train_timesteps
        if self.num_inference_steps <= 1:
            return [train_steps - 1]
        steps = torch.linspace(train_steps - 1, 0, self.num_inference_steps, dtype=torch.long)
        unique_steps: list[int] = []
        seen = set()
        for v in steps.tolist():
            if v not in seen:
                unique_steps.append(v)
                seen.add(v)
        return unique_steps

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
        cond: torch.Tensor | None = None,
        uncond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        _ = uncond
        z_t = torch.randn(shape, device=device, dtype=dtype)
        timesteps = self._inference_timesteps()
        bsz = shape[0]

        for idx, step in enumerate(timesteps):
            t = torch.full((bsz,), step, device=device, dtype=torch.long)
            pred_noise = _predict_noise_with_cfg(
                model=model,
                z_t=z_t,
                t=t,
                cond=cond,
                guidance_scale=guidance_scale,
            )

            alpha_bar_t = self.scheduler.alpha_cumprod[step].to(device=device, dtype=dtype)
            if idx + 1 < len(timesteps):
                prev_step = timesteps[idx + 1]
                alpha_bar_prev = self.scheduler.alpha_cumprod[prev_step].to(device=device, dtype=dtype)
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device, dtype=dtype)

            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            pred_x0 = (z_t - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t

            sigma_t = self.eta * torch.sqrt(
                ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * (1.0 - alpha_bar_t / alpha_bar_prev)
            )
            dir_coeff = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t * sigma_t, min=0.0))
            mean = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_coeff * pred_noise

            if step > 0 and self.eta > 0.0:
                z_t = mean + sigma_t * torch.randn_like(z_t)
            else:
                z_t = mean

        return z_t
