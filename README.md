# NULLXES KURONO

Multimodal video + audio generation engine blueprint (DiT-first, MoE-ready).

## Owner

- Organization: `NULLXES LLC`
- Author: `NULLXES`
- Contact: `ceo@nullxes.com`

## Goals

- Build a clean-room multimodal stack with staged training on H200.
- Start dense for stability, then migrate to sparse MoE.
- Keep two product lines:
  - `RAW`: research / uncensored experimentation branch.
  - `PROD`: safety-guarded branch for end users.

## Selected Baseline Components (current)

- Video VAE: `WF-VAE` (primary candidate).
- Video backbone (S1): `Latte`-style latent DiT baseline.
- Production-scale references: `Open-Sora` / STDiT family.
- MoE implementation target: MegaBlocks-style dropless MoE.
- Audio tokenizer target: DAC-style continuous latents.

## Repo Layout

```text
KURONO/
  configs/
  core/
    video_dit/
    audio_dit/
    moe/
    cross_modal/
  docs/
```

## Stage Plan

- S1: Video-only dense baseline (`WF-VAE` + latent DiT), metrics pipeline.
- S2: FFN -> MoE swap in selected blocks, routing telemetry.
- S3: Audio branch + joint AV token space, temporal alignment.
- S4: RAW/PROD split + safety/watermark + distillation.

## Next Immediate Work

1. Import `WF-VAE` encode/decode adapter into `core/video_dit`.
2. Add minimal train config for 256px short clips on H200.
3. Add evaluation harness: FVD, CLIP-score, temporal consistency probes.

## S1 Structural Run (No HF Download)

Install:

```bash
pip install -r requirements.txt
```

PowerShell:

```powershell
./run_s1.ps1 -Steps 50 -BatchSize 1 -Frames 65 -Height 256 -Width 256 -Precision bf16 -Device cuda
```

Bash:

```bash
bash run_s1.sh
```

Direct Python:

```bash
python train_s1.py --steps 50 --batch-size 1 --frames 65 --height 256 --width 256 --precision bf16 --device cuda --mock-vae
```

Note:
- `--mock-vae` keeps this run fully local and structure-only.
- Without `--mock-vae`, you must provide `--wfvae-pretrained` pointing to a local WF-VAE export (e.g. `config.json` + `*.ckpt`).

