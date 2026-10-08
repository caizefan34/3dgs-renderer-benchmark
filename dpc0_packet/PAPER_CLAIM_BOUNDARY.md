# What would make DPC-GS publishable?

## Not enough
- porting VkSplat components to CUDA;
- replacing atomics by a warp reduction;
- reporting a faster backward kernel only;
- deterministic gradients at a large speed penalty.

## Potentially strong paper
A system whose central contribution is the joint property:

    reproducible training state transition
    + primitive-centric race-free backward
    + high performance

Target evidence for the eventual paper track:
1. Same build/input/seed on one A100 -> bitwise-identical gradients and checkpoints.
2. Same build/input/seed across multiple A100s -> preferably bitwise-identical trajectory.
3. Full 30K room/garden/bicycle repeated runs -> identical topology/checkpoint hashes.
4. >=3.0x end-to-end wall speed vs frozen C0, with held-out quality within strict tolerance.
5. Head-to-head speed/reproducibility comparison with VkSplat and DISTWAR where protocol allows.
6. Ablation of:
   - primitive-centric tile partials;
   - deterministic segmented reduction;
   - deterministic densification/statistics;
   - projection-backward/optimizer fusion.

If DPC reaches only the known speed recipe without the reproducibility result, originality is weak.
If it reaches reproducibility but is much slower than C0, systems impact is weak.
The paper needs both.
