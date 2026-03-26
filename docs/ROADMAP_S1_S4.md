# KURONO Roadmap S1-S4

## S1 - Dense Video Baseline

Objective:
- Prove stable training and acceptable temporal coherence.

Deliverables:
- WF-VAE adapter for encode/decode.
- Dense latent DiT training loop.
- Eval: `eval_s1.py` (recon MSE + temporal L1 on data); FVD + CLIP + full consistency bundle still TODO for prod eval.

Exit criteria:
- Reproducible training run with fixed seeds.
- Baseline checkpoint and evaluation report.

## S2 - MoE Upgrade

Objective:
- Increase model capacity without linear compute growth.

Deliverables:
- FFN -> MoE replacement in selected transformer blocks.
- Router losses and telemetry dashboard outputs.
- Ablation: dense vs MoE on same data slice.

Exit criteria:
- Quality not regressing beyond threshold.
- Better throughput/quality tradeoff than dense baseline.

## S3 - Audio + AV Joint Modeling

Objective:
- Add audio generation and alignment with video timeline.

Deliverables:
- Audio latent encoder/decoder integration.
- Audio DiT baseline.
- Joint AV block with temporal alignment module.

Exit criteria:
- Audible quality baseline + measurable AV sync gain.

## S4 - Productionization Split

Objective:
- Maintain research freedom and production safety simultaneously.

Deliverables:
- RAW and PROD config branches.
- Safety filter stack and watermark integration for PROD.
- Distilled inference tier for cost/latency control.

Exit criteria:
- Separate release pipelines for RAW and PROD.
- Policy-compliant production inference path.

