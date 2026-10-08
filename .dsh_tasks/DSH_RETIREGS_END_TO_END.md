# DSH GOAL — RETIREGS END-TO-END EXPERIMENT PROGRAM

## Mission
Complete the RetireGS research program in C:\Users\36570\3dgs-renderer-benchmark and on MX/A100.
RetireGS hypothesis: after densification stops, a large subset of Gaussians can cease being optimization variables while remaining part of the rendered representation; if this convergence headroom exists, exploit it with a certified depth-aware compositing cache so retired Gaussian work is removed from repeated training computation.

This is a fail-closed research task. Do not rescue failed gates, lower thresholds after seeing results, or invent candidate variants to force a positive outcome.

## Existing state — DO NOT restart blindly
Local repo: C:\Users\36570\3dgs-renderer-benchmark
MX host alias: mx
Remote repo: ~/3dgs-renderer-benchmark
RetireGS root: /mnt/storage_pool/liaoyuanjun/retiregs
Existing prereg: /mnt/storage_pool/liaoyuanjun/retiregs/RETIRE0_PREREG.json
Existing source runner: /tmp/retire0_source_runner.py
Existing continuation harness: /tmp/retire0_continue.py
Existing room source output: /mnt/storage_pool/liaoyuanjun/retiregs/RETIRE0_room_source
Existing room source PID was 660596 on GPU7 when last checked.
SOTA multiphase is separately running on GPU6. Do not disrupt it.

Before acting, inspect current state and resume from evidence already present. Never launch duplicate source runs if valid ones are still active or completed.

## Environment contract
Authoritative training cohort:
- A100-PCIE-40GB, sm_80
- torch 2.7.1+cu118
- gsplat 1.5.3
- CUDA runtime 11.8
- explicit CUDA 11.8 compiler path if compilation is needed
- nerficg Python environment currently used by the reference_v1 trainer
## Phase A — Audit and source integrity
1. Verify RETIRE0_PREREG.json parses and preserve its thresholds exactly.
2. Verify room source checkpoints include model state + optimizer_state at 15K and 30K.
3. Verify topology row identity is stable after 15K: Gaussian count/index semantics must not change from 15K to 30K.
4. Verify the continuation harness uses the same 15K model, optimizer state, and frozen camera sequence for all paired variants.
5. Verify retirement semantics: retired rows have gradients zeroed AND Adam exp_avg/exp_avg_sq zeroed; post-run max absolute parameter drift for retired rows must be exactly 0 within serialization/numerical representation.
6. Record source/provenance hashes. If source artifacts are corrupted or semantically invalid, mark BLOCKED_SOURCE_INTEGRITY and stop rather than silently repairing scientific meaning.

## Phase B — RETIRE0 room oracle
Use future-information oracle ranking based on real 15K→30K parameter change. Test paired continuation:
- 0% retirement control
- 25%
- 50%
- 70%
- 80%
- 90%

RETIRE0 is a convergence oracle only. It still computes full forward/backward, so NEVER claim speedup from Phase B.

Pilot PASS gate, frozen before results:
- minimum retirement fraction: 50%
- |delta PSNR| <= 0.15 dB versus paired 0% control
- |delta SSIM| <= 0.005 versus paired 0% control
- retired parameter max_abs_drift = 0

If 50% retirement fails either quality gate, verdict RETIRE0_ROOM_FAIL and close the RetireGS retirement lane. Do not implement caching, CUDA, or rescue variants.

If 50% passes, continue upward to identify the maximum passing retirement fraction among 50/70/80/90.
