# MSB0-P0 Raw Trace Index

All experimental work executed on **EPIC-05** (remote-guard compliant). Local
Windows = control-plane only (inspection, transfer, small-JSON aggregation).

## Execution environment (per REMOTE_ONLY_GUARD.md)
- Host: `dsw-933622-64cc845c7d-lqswc` (via `ssh EPIC-05`, 8.130.30.251:1024)
- GPU: NVIDIA A100-SXM4-80GB (CUDA_VISIBLE_DEVICES pinned)
- Python: `/root/miniforge3/envs/p16_fastergs/bin/python` (torch 2.8.0+cu128, pycolmap 4.2.0)
- MSB0 workspace: `/root/msb0_p0/{scripts,out,trace}`
- Provenance record: `trace/REMOTE_GUARD_PROVENANCE.md`

## Source lock (Gate 1)
- faster-gaussian-splatting @ `ae665af2f91a4973bb92c40fdc4cba8d12fe3c96` (2026-09-19)
- nerficg @ `e458ac6e41810ad6d9405687c1e698fadd28dac5` (2026-09-06)
- Kernels: `blend_cu` (kernels_forward.cuh:355-456), `blend_backward_cu` (kernels_backward.cuh:252-475)

## Raw per-scene replay traces (EPIC-05: /root/msb0_p0/out/)
| File | Scene | Content |
|------|-------|---------|
| `msb0_replay_room.json` | room (image id 0, DSCF4667.png) | M0/M1/M2/M3 fidelity + M0_vs_autograd; tile [92,1], 1168 positions, 37 buckets |
| `msb0_replay_garden.json` | garden (DSC07956.png) | same; tile [59,28], 2048 positions, 64 buckets |
| `msb0_replay_bicycle.json` | bicycle (_DSC8679.png) | same; tile [32,41], 2048 positions, 64 buckets |

Each replay JSON contains, per mode: `dL_dcolor`, `dL_dalpha`, `dL_dconic`,
`dL_dmean2d` with `{cos, rel_l2, max_abs_err, nan_inf}`; plus `gate4_pass_M0`
and `saved_state_bytes`.

## Raw per-scene economics traces
| File | Scene | Content |
|------|-------|---------|
| `msb0_economics_room.json` | room | full-image saved-state bytes; M1/M2/M3 reduction |
| `msb0_economics_garden.json` | garden | same |
| `msb0_economics_bicycle.json` | bicycle | same |

## Strong-backend timing source (input, not produced here)
- `fgs_strong_baseline_final/FGS_TIMING.json` (fused lane): room 260 s/iter,
  garden 523 s/iter, bicycle 448 s/iter; peak_vram_allocated 8.96/11.12/10.87 GB.

## Aggregation / replay scripts (EPIC-05: /root/msb0_p0/scripts/, mirrored locally)
| Script | Role |
|--------|------|
| `msb0_oracle.py` | projection, `composite_tile`, `backward_tile` (M0/M1/M2/M3), `autograd_ground_truth`, `fidelity_metrics`, `passes_gate4`. All float64. |
| `msb0_run.py` | COLMAP readers, frozen-camera (image id 0) selection, `run_scene`, `main`. CLI: `--scene --ckpt --sparse --out --max-positions`. |
| `msb0_economics.py` | full-image per-tile bucket count, saved-state bytes, per-candidate reduction. |
| `dump_results.py` | prints fidelity per scene/mode. |
| `run_econ.sh` | driver: runs economics for all three scenes sequentially. |

## Aggregated deliverables (this directory)
- `MSB0_VERDICT.json`
- `MSB0_REPORT.md`
- `MSB0_SOURCE_LOCK.json`
- `MSB0_DEPENDENCY_GRAPH.md`
- `MSB0_ALGEBRA.md`
- `MSB0_REPLAY_FIDELITY.json`
- `MSB0_ECONOMICS.json`
- `MSB0_PRIOR_ART.md`
- `MSB0_TRACE_INDEX.md` (this file)
- `MANIFEST.sha256`
