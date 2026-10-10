# C51J — CVPR27-C51-JOINT Evidence & MX Validation (Index)

Execution-agent deliverable set. Platform: **MX / NVIDIA A100-PCIE-40GB** (reachable via SSH host `mx`, `bms-39468022-001`).
Formal output dir: `/mnt/storage_pool/liaoyuanjun/CVPR27_C51_JOINT_MX/` (created; lightweight audit copies staged in `audit/`).
Local delivery dir: `C51J_DELIVERY/`.

> **Current verdict:** `BLOCKED_NEEDS_EXCLUSIVE_GPU` — all P0–P3 + P6 work is COMPLETE and verified; P4/P5 training is blocked on an external resource condition (no A100 with ≥30 GiB free) AND on a Codex candidate manifest (per the objective's "execute P4–P5 only after Codex candidate verification passes"). No results were fabricated.

## Deliverables (all SHA256-pinned in `MANIFEST_SHA256.json` / `.txt`)

| File | Phase | Status | What it is |
|------|-------|--------|-----------|
| `C51J_BASELINE_EVIDENCE_AUDIT.md` | P1 | DELIVERED | 1.164× = **historical aggregate only**; per-run raw + dedicated protocol file NOT located; gsplat-combination (not HiGS native); recoverable provenance pinned (fastgs@`d4d33b6`, speedy-splat@`b9dd42d1`, paper-protocol@tco0_work, gsplat commit `77ab983`) |
| `C51J_C51_CAUSAL_AUDIT.md` | P2 | DELIVERED | Stage-5 four-arm (A/B/C/D) cost table; R0.2 gradient Top-K failure (prev-iter 0.528 ≈ random 0.500); strict protocol boundary vs 132-job |
| `C51J_WORKLOAD_COST.csv` | P2 | DELIVERED | 81-row real cost table (component ms, population trajectory, R0.2 ceilings), tagged by source stage |
| `C51J_FROZEN_PROTOCOL.json` | P3 | DELIVERED | Frozen executable protocol: 4 arms, split isolation, timing+VRAM contract, gates; MX env fingerprint |
| `C51J_harness.py` | P3/P4 | DELIVERED (turnkey) | Execution-gated training harness: storage home-fallback, per-GPU free-mem gate, VRAM capture, split isolation; `--emit-plan` prints the P4 command matrix; py_compile OK; gate verified live |
| `C51J_harness_splittest.py` | P3 | DELIVERED (test PASS) | Unit test proving train/val/heldout disjoint+exhaustive; assert_train_only raises on leak |
| `C51J_P4_EXECUTION_PLAN.txt` | P4 | DELIVERED | Copy-pasteable runbook: 1K smoke / paired 5K / confirmatory seeds × room,bicycle,garden × 3 arms + P5 gate criteria |
| `C51J_1K_SMOKE.json` | P4 | NO_DATA (blocked) | Schema-only placeholder; no smoke ran |
| `C51J_5K_PAIRED.csv` | P4 | NO_DATA (blocked) | Header-only; no paired 5K ran |
| `C51J_QUALITY_PERFORMANCE.csv` | P4 | NO_DATA (blocked) | Header-only; no new quality metrics |
| `C51J_FINAL_VERDICT.json` | all | DELIVERED | `BLOCKED_NEEDS_EXCLUSIVE_GPU`; P4-readiness validated; gate rationale recorded |
| `RUN_STATUS.json` | all | DELIVERED | Per-file status + resource state + round log |
| `MANIFEST_SHA256.json` / `.txt` | P6 | DELIVERED | SHA256 of every delivery file |

## Verified environment (this agent's account, `liaoyuanjun`)
- Interpreter: `/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python` — torch 2.7.1+cu118, CUDA 11.8, **gsplat 1.5.3**, cxx11_abi=1, simple_knn OK.
- Default `python3` has a BROKEN torch (do not use). Root `gsplat` env is inaccessible (Permission denied) — not needed.
- R28-B ABI failures are confined to prebuilt HiGS/FasterGS rasterizers (C++ ABI); gsplat 1.5.3 imports cleanly → does NOT block a fresh gsplat experiment.

## Verified blocker (as of last probe, 2026-10-10)
1. **GPU free memory** — all 8 A100s host other users' jobs; max free seen ≈18.3 GiB (< 30 GiB gate). OOM is a P5 disqualifier, so the gate is deliberately NOT lowered without a measured VRAM floor.
2. **Codex manifest** — no C51 candidate manifest has arrived; P4 begins only after Codex verification per the objective.
- Storage pool is 100% full (`data/`=2.1 TB, user-owned, protected) but this is NOT a hard blocker: the harness falls back to writable home (`/home/liaoyuanjun`, 27 GB) / `/dev/shm` (200 GB).

## Exact unblock procedure (one command, when both conditions clear)
```bash
# On MX, with an A100 showing >=30 GiB free:
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
cd /home/liaoyuanjun/3dgs-renderer-benchmark
$PY C51J_harness.py --arm native_baseline --scene room --iters 1000 --gpu <FREE_GPU> --repo-root $PWD
# then the rest of the matrix via:  $PY C51J_harness.py --emit-plan
```
Priority scenes: **room, bicycle, garden**. Compare candidate vs `native_baseline` vs `opacity_pruning_control` (Speedy-Splat `prune(prune_ratio)` @ `b9dd42d1`). Collect `runs/*.json` into the P6 CSVs, then judge `GO_TO_30K` against the P5 criteria (wall ≥1.10×, ΔPSNR ≥−0.10, ΔSSIM ≥−0.002, ΔLPIPS ≤+0.005; no NaN/OOM/illegal lineage).

## Integrity
- No unique experiment data deleted or moved; all MX staging double-ended SHA256 verified.
- No 132-job recomputation claimed; no C51↔132-job protocol merge; no aggregate hiding per-scene.
- All new outputs written to the independent `CVPR27_C51_JOINT_MX/` dir and local `C51J_DELIVERY/`.
