# CGT-Splat research prototype

This package contains a Chinese research proposal and a runnable CPU reference.
It is a proposed representation, not a validated replacement for 3D Gaussian Splatting.

- Read `CGT_Splat_研究方案.md` for definitions, prior-art boundaries, costs, and rejection gates.
- Run `cgt_reference.py` to reproduce algebra checks and synthetic fits.
- Read `results/results.json` for raw results, optimization termination status, and limitations.
- View `results/synthetic_probe.png` for the successful smooth case and failed occlusion case.
- The `.npz` files contain the fitted synthetic models from the reported run.

Requires Python 3.10+, numpy, scipy, and matplotlib.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cgt-mpl \
python3 cgt_reference.py --output results --iterations 500
```

Evidence status: algebra, affine camera-chart restriction, and analytic derivatives numerically checked;
limited synthetic compression demonstrated; fine interleaved occlusion failed at the tested budgets.
All nine optimization fits reached the 500-iteration limit. No convergence optimum is claimed.
No real 3D scenes, spatial hierarchy, CUDA kernels, or end-to-end speed/energy measurements are included.
Component-count reduction is not a GPU speedup. Novelty relative to all prior work is unresolved.

The exact mixture composition grows the number of components multiplicatively. Its usefulness
depends on successful cost-effective recompression, which has not been established for real scenes.
