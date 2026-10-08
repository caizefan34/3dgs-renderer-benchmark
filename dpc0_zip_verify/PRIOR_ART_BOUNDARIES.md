# DPC0 PRIOR-ART / PAPER BOUNDARY — frozen 2026-09-30

DPC0 is NOT allowed to claim the following as novel by themselves:

1. Per-Gaussian / primitive-centric backward.
   - VkSplat (Eurographics 2026) already includes per-Gaussian and tensor-based rasterization
     backward passes that eliminate pixel-level atomic contention.
   - 3DGART (arXiv 2026) reorganizes ray-traced Gaussian backward around primitives and reports
     roughly 3–3.5x raw training speedup over a per-pixel baseline.

2. Atomic reduction optimization for rasterized 3DGS.
   - DISTWAR already applies serialized and butterfly warp-level atomic reduction to the original
     3DGS backward and reports about 2x training speedup in its public implementation.
   - Earlier public rasterizers also use warp/shared reductions before atomics.

3. Exact tile culling, projection-backward/Adam fusion, or fully-fused training by themselves.
   - VkSplat already reports exact scan-line tile culling, per-Gaussian backward, fused projection
     backward + Adam, fused loss, >3.3x end-to-end speed, and lower VRAM.

References:
- VkSplat: https://harry7557558.github.io/vksplat/
- VkSplat arXiv: https://arxiv.org/abs/2605.00219
- 3DGART: https://arxiv.org/abs/2608.17298
- DISTWAR public implementation:
  https://github.com/Accelsnow/gaussian-splatting-distwar
- NVIDIA CCCL determinism:
  https://nvidia.github.io/cccl/unstable/cccl/determinism.html
  https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/

Observed field gap motivating DPC:
public Gaussian-SLAM documentation explicitly notes that the differential Gaussian rasterizer remains
non-deterministic even when the rest of the system is made deterministic:
https://github.com/vladimiryugay/gaussian-slam

Candidate contribution boundary to test:

  A deterministic primitive-centric backward and training state-transition pipeline for rasterized
  3DGS that removes floating-point atomic race ordering while retaining competitive or superior
  throughput.

A future paper claim is authorized ONLY if the work demonstrates:
- actual run-to-run training reproducibility, not just deterministic micro-reduction;
- correct gradients / equivalent quality;
- meaningful speed, ideally >=3x end-to-end vs frozen C0;
- comparison against VkSplat/DISTWAR and other fast baselines.

Do NOT use "first" until a fresh submission-time literature audit confirms it.
