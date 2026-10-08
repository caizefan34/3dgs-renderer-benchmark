# REPRODUCIBILITY — Commands

**Generated:** 2026-10-02  |  Inventory of canonical entry points. Exact remote GPU commands live in the *_SOURCE_LOCK.json / pack READMEs; do NOT re-run GPU work as part of this support pack.

## Build (gsplat extension, EPIC-05 remote)
```
# gsplat-b1-clean-v153 fork is the canonical renderer; build the CUDA extension there.
# Per-pack build scripts (remote) compile the .cu kernels via ninja/build.ninja.
# Example (TFC1): _tfc1_kernel/build.ninja + tfc1_build.py  (local smoke artifacts only)
python -c "import torch,gsplat;print(torch.__version__, torch.version.cuda, gsplat.__version__)"
```

## Benchmark / render (renderer-local: TFC, DFC)
```
# TFC benchmark (renderer-local, frozen PLY) — canonical entry: tfc1_benchmark.py / TFC2_BENCHMARK
python tfc1_benchmark.py --scene room --mode ellipse   # mode1 = ellipse-exact candidate
# DFC per-scene E2E: _dfc2_pack/dfc2_run.py (assembler self-test PASS)
python _dfc2_pack/dfc2_run.py --selftest
```

## Training (full 30K)
```
# K50-B1 canonical 30K (C51_STAGE4B, A100-80GB): use the C51 training harness (stage4b report Section 1 config).
# Strong baselines: python strong_baseline_train.py --baseline speedy-splat --scene bicycle --method native --gpu 0
# DFC_FC0 (PENDING, REQUIRES_VALIDATION): _dfc2_pack/train_dfc.py (unverified scaffold; 200-iter smoke on room REQUIRED before 30K).
```

## Evaluation
```
python strong_baseline_eval.py ...          # baseline quality eval
python c42_13scene_eval.py                  # 13-scene eval
python test_eval_canonical.py               # canonical eval check
# K50-B1 gate eval: reports/phase-c51-stage4b.md Section 3-4 (Gate A cosine, Gate B speedup)
```

## Timing-ledger assembly (CPU)
```
python _pub0_ledger/build_pub0.py     # rebuild PUB0 timing ledger from existing artifacts
python _pub0_ledger/build_report.py
```

## Prohibitions
- Do NOT run K50 canonical revalidation (Task 5 prereg only; activate only after PUB0).
- Do NOT run TFC_FC0 / DFC2 / DFC_FC0 as part of this support pack (current GPU work has priority).
- Do NOT merge cohorts or convert renderer-local speedups into training speedups.
