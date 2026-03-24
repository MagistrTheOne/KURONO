# KURONO Architecture (v0.1)

## 1) Latent Stack

- Video latent path:
  - `video -> WF-VAE encoder -> z_v`
  - target shape guideline: `[B, C, T/4, H/8, W/8]`
- Audio latent path:
  - `audio -> audio codec encoder -> z_a`
  - continuous latent space preferred for diffusion.

## 2) Core Backbone

- Unified diffusion transformer interface:
  - `VideoDiTBackbone`
  - `AudioDiTBackbone`
  - `JointAVBackbone` (later stage)
- Initial training in dense FFN mode for stability.
- Planned MoE injection point: FFN sublayers only.

## 3) MoE Strategy

- Start with `top-2` routing.
- Add `aux loss` for load balance and `z-loss` for router stability.
- Track per-layer expert utilization and router entropy.
- Keep dropless routing as the target behavior.

## 4) Cross-Modal Sync

- Shared temporal index mapping between audio and video token groups.
- RoPE/positional alignment with explicit time-axis agreement.
- Keep implementation modular (no hard coupling to one paper codebase).

## 5) Product Split

- RAW branch:
  - minimal censorship, research velocity.
- PROD branch:
  - policy filters, watermarking, safer decoding defaults.
  - reproducibility metadata enforced (`seed`, sampler, cfg, steps, model hash).

## 6) Runtime and Deployment Notes (H200-first)

- Preferred precision:
  - train: BF16 first, then selective FP8.
  - infer: BF16/FP8 based on quality tier.
- Parallelism targets:
  - data + tensor + expert parallel.
- Observability:
  - generation latency, memory, expert load, failure traces.

