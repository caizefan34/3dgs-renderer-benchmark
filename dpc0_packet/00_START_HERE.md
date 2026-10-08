# DPC0 — Deterministic Primitive-Centric 3DGS Backward

## Why this is not just a VkSplat copy

VkSplat already proves that a system of per-Gaussian backward, exact culling and fused optimizer can
exceed 3.3x over a CUDA+PyTorch baseline. DPC does not claim those ingredients.

DPC asks a different question:

> Can rasterized 3DGS training be made race-free and reproducible while retaining high performance?

The current project repeatedly encountered CUDA atomic accumulation noise in H0, UCR0, MPI0 and ELA0.
DPC0 directly attacks that common root.

If DPC0 passes:
- DPC1 builds the fast deterministic full state transition.
- DPC2 is the actual publication gate: >=3x full training + bitwise repeated trajectories + quality.

If DPC0 is deterministic but slow, do not hide it; return DPC0_TOO_SLOW.
