# VRIR0 Oracle Patch Allowlist

Allowed repository changes during VRIR0:
- new `experiments/vrir0/**` or equivalent oracle-only Python/config/report files;
- minimal Python hook in canonical trainer, guarded by an explicit oracle flag and behaviorally inert when disabled;
- provenance/evidence outputs outside tracked source code.

Forbidden:
- renderer CUDA/C++ code;
- installed gsplat binary/source modification;
- baseline training hyperparameter changes;
- AABS0 or FGS-H0 source/result modification;
- changes to camera sequence, seed, dataset, optimizer, loss or densification schedule.

DSH must save `git diff --stat` and full diff in provenance evidence before and after each run.
