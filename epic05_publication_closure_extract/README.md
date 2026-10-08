# EPIC-05 publication-closure campaign

This bundle defines the remaining **experiment-only** work for the paper on EPIC-05
(8× NVIDIA A100-SXM4-80GB). It does **not** reopen C0 optimization. It exists to
strengthen reproducibility, attribution, seed robustness, and H8 mechanism/generality.

## Required stages

| Stage | New work | Purpose |
|---|---:|---|
| P0 | 16 short 2K calibration runs | verify all eight physical GPUs are comparable before publication timing |
| P1 | 26 × 30K | C0 V3 vs B1A, 13 scenes, seed 42 — cross-system reproduction |
| P2 | 39 × 30K | A0/A1/A2 on 13 scenes; C0 reused from P1 — repeat cumulative attribution on SXM4 |
| P3 | 6 × 30K | missing eps2d corners on room/bicycle/garden; base corners reused from P1 |
| P4 | 16 × 30K | B1A/C0 × seeds 43/44 × {drjohnson, train, bicycle, room}; seed 42 reused |
| P5 | microbench | H8 off/on direct-backward correctness + timing on room/bicycle/garden |
| P6 | microbench | H8 generality validation in pristine gsplat/isolated drop-in path; no C0 production changes |

Total **new full-training runs = 87**.

## Why these and no others

Existing evidence already closes full-training correctness, matched baselines,
13-scene ablation, eps2d sensitivity, external-system context, and multi-seed on the
original A100-PCIE-40GB system. The remaining experimental weakness is hardware/system
scope and whether the H8 mechanism survives outside the one production composition.
Therefore EPIC-05 is used to *replicate the decisive evidence*, not to search for H9/H10.

EPIC-05 is still SM80/A100. A positive EPIC result establishes **cross-system /
cross-board reproducibility**, not cross-architecture generalization. The paper must
continue to say A100-scoped unless a non-A100 experiment is later added.

## Frozen pairing policy

All jobs belonging to one comparison group execute sequentially on the **same physical
GPU**. This prevents 8-GPU board-to-board variance from leaking into speedup ratios.

For seed-42 scene groups the order is:
`B1A -> A0 -> A1 -> A2 -> C0`, plus the two eps2d missing corners on the 3 sensitivity scenes.
For seeds 43/44, `B1A -> C0` stays on one GPU.

Do not merge these numbers into the original PCIe cohort. Report them as an independent
EPIC-05 A100-SXM4-80GB replication cohort.

## Gates

P0:
- exactly 8 `A100-SXM4-80GB`;
- record driver, clocks, power, temperature and UUIDs;
- same Python/PyTorch/CUDA cohort if possible;
- 2K room calibration must have <=5% max relative deviation from the per-arm median.
If this fails, exclude abnormal GPUs or restrict all publication pairs to the validated set.

P1:
- all 26 runs valid and uncontaminated;
- primary: geometric mean `wall_B1A / wall_C0`;
- report `n_faster / 13`, per-scene ratios and quality deltas;
- do **not** require the SXM4 ratio to equal 1.0685×.

P2:
- compare A0→A1→A2→C0 within the same rebuilt C0 binary family;
- report the SXM4 ladder separately from the PCIe ladder;
- never use it to retroactively attribute the 1.0685× headline.

P3:
- reproduce the eps2d 2×2 on SXM4 and compute a hardware-specific timing/quality bound.

P4:
- all 12 per-scene seed ratios should be shown, not only the geomean;
- drjohnson remains a per-scene trajectory effect, never a quality-improvement claim.

P5:
- exactness first, timing second;
- preserve the production H8 patch SHA `8fb23acca984...`;
- direct backward and F+B must be reported separately.

P6:
- validation only. Implement the same sufficient-statistic factorization in a pristine
  gsplat backward or isolated drop-in path;
- required output: gradient equivalence + 3-scene microbench;
- no 30K convergence run is required unless the transformation changes training semantics.

## Execution

1. Connect to EPIC-05 and update/clone the repository under `/root/3dgs-renderer-benchmark`.
2. Run `preflight_epic05.sh`.
3. Recreate the frozen B1A/C0/publication trainer under the EPIC runtime. Record source,
   patch and build provenance. Rebuilt `.so` hashes may differ; do not silently substitute
   a different source tree.
4. Run the 8-GPU calibration with `calibrate_8gpu.py`.
5. Only after P0 PASS, launch `run_group_scheduler.py` with `epic05_manifest.json`.
6. Run `analyze_epic05.py` after all 87 full-training jobs are complete.
7. Run P5 and P6 as separate microbench evidence trees.
8. Freeze an `EPIC05_FINAL_VERDICT.json`; never overwrite the original FINAL-30K artifacts.

Example scheduler invocation:

```bash
python run_group_scheduler.py \
  --manifest epic05_manifest.json \
  --trainer /root/3dgs-renderer-benchmark/publication_trainer.py \
  --python /root/miniforge3/envs/gsplat/bin/python \
  --out /root/3dgs-publication-epic05/full30k \
  --gpus 0,1,2,3,4,5,6,7
```

## Manuscript consequence

If P1 is positive, wording may be upgraded to:
“the complete-system speedup is reproducible across independent A100 PCIe-40GB and
A100-SXM4-80GB systems, with hardware-dependent magnitude.”

It must **not** become “cross-GPU-architecture generalization.”
