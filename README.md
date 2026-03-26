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

## Training corpora (planned, Hugging Face)

These datasets are the **intended** sources for KURONO training and curation (licenses and column schemas differ per card — review on Hugging Face before use).

| Corpus | Hugging Face |
|--------|----------------|
| Panda-70M | [multimodalart/panda-70m](https://huggingface.co/datasets/multimodalart/panda-70m) |
| Valley WebVid2M pretrain (703K) | [luoruipu1/Valley-webvid2M-Pretrain-703K](https://huggingface.co/datasets/luoruipu1/Valley-webvid2M-Pretrain-703K) |
| VideoUFO | [WenhaoWang/VideoUFO](https://huggingface.co/datasets/WenhaoWang/VideoUFO) |
| OpenVid-1M | [nkp37/OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) |
| Civitai top SFW (images + metadata) | [wallstoneai/civitai-top-sfw-images-with-metadata](https://huggingface.co/datasets/wallstoneai/civitai-top-sfw-images-with-metadata) |

**Current code path:** `train_s1.py` / `VideoMP4Dataset` expect **local** video files (or a JSON manifest from `data/filter_dataset.py`), not a live `datasets` stream. Staging options: materialize HF splits to disk (e.g. `huggingface-cli download` / your own export) into a tree of `.mp4`/`.webm`/… then point **`--data-path`** at it, or generate a **`--filter-manifest`** from your metadata. A dedicated HF-iterator loader is not in-tree yet.

**Note:** the Civitai entry is **image**-oriented; it does not plug into the current **video** clip loader as-is. Use it for image pipelines, captions/metadata, or later multimodal stages — not for raw `VideoMP4Dataset` without an adapter or conversion step.

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

1. ~~WF-VAE encode/decode via `core/video_dit/wfvae_adapter.py` + `core/vae` (supports `diffusion_pytorch_model.safetensors` or `*.ckpt`).~~
2. ~~Minimal S1 recipe YAML: `configs/kurono.s1_256.yaml` (reference; `train_s1.py` still uses CLI).~~
3. ~~Lightweight probes: `eval_s1.py` (reconstruction MSE + temporal L1).~~ Add FVD / CLIP-score with separate deps on prod eval machines.

## WF-VAE weights (production only)

Weights are **not** downloaded by this repo or CI. Ship them only on training/inference hosts.

- **Reference weights (16-ch, matches DiT `latent_channels=16`):** [chestnutlzj/WF-VAE-L-16Chn](https://huggingface.co/chestnutlzj/WF-VAE-L-16Chn) — copy the snapshot directory (e.g. `config.json` + `diffusion_pytorch_model.safetensors`) to the machine, then pass `--wfvae-pretrained /path/to/that/folder`.
- **Upstream code / paper:** [PKU-YuanGroup/WF-VAE](https://github.com/PKU-YuanGroup/WF-VAE).

Local dev can use `--mock-vae` without any VAE files.

## S1 Structural Run (no weight download in-repo)

Install:

```bash
pip install -r requirements.txt
```

PowerShell (requires video data):

```powershell
$env:KURONO_DATA_PATH = "D:\path\to\video_or_folder"
./run_s1.ps1 -Steps 50 -BatchSize 1 -Frames 65 -Height 256 -Width 256 -Precision bf16 -Device cuda
# or: ./run_s1.ps1 -DataPath "D:\path\to\video_or_folder" ...
```

Bash:

```bash
export DATA_PATH=/path/to/video_or_folder   # or KURONO_DATA_PATH
bash run_s1.sh
```

Direct Python:

```bash
python train_s1.py --steps 50 --batch-size 1 --frames 65 --height 256 --width 256 --precision bf16 --device cuda --mock-vae --data-path /path/to/video_or_folder
```

Cheap eval on one or more batches (optional):

```bash
python eval_s1.py --data-path /path/to/video_or_folder --mock-vae
```

Note:

- Training always needs **`--data-path`** or **`--filter-manifest`** (scripts set `DATA_PATH` / `-DataPath` / `KURONO_DATA_PATH`).
- `--mock-vae` = structure-only VAE; no weight files.
- Real VAE: **`--wfvae-pretrained`** = local directory with **`config.json`** plus **`diffusion_pytorch_model.safetensors`** (HF layout) or **`*.ckpt`**.

