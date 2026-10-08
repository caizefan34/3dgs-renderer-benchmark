# PGR0 Oracle Patch Allowlist

Allowed repository changes during PGR0:
- new `pgr0_oracle/**` oracle-only Python/config/report/artifact files;
- the CPU reference reimplementation and analyzer under `pgr0_oracle/scripts/`;
- provenance/evidence outputs under `pgr0_oracle/artifacts/`.

Forbidden:
- renderer CUDA/C++ code (`.cu`, `.cuh`, `.cpp`, `.cc`, `.cxx`);
- installed gsplat binary/source modification;
- baseline training hyperparameter changes;
- C0, AABS0, FGS-H0, H8, or rasterizer source/result modification;
- camera sequence, seed, dataset, optimizer, loss, or densification changes;
- any CUDA build or kernel run in P0.

DSH must save `git diff --stat` and full diff in provenance evidence before and
after each run.
