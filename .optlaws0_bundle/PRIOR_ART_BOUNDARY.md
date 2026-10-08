# OPTLAWS0 Prior-Art Boundary

Frozen date: 2026-09-29

Known direct boundaries:

- HarmoGS (2026): cross-view gradient conflict in 3DGS and dual-view
  conflict-aware gradient harmonization. Therefore "gradient conflict exists"
  is not new.

- Efficient Multi-view Training for 3DGS (2025): identifies high variance in
  single-view stochastic gradients and reduces it by multi-view training.
  Therefore "single-view gradients are noisy" is not new.

- Two-View Accumulation as the Primary Training Lever (2026): shows simple
  two-view accumulation can dominate more elaborate pairing/surgery in a
  hybrid-capture setting. Therefore "two gradients average better than one"
  is not new.

- S2Gaussian (CVPR 2025): uses historical per-Gaussian gradient trend /
  direction consistency. Therefore historical gradient state itself is not new.

Potentially new empirical laws under this bounded audit:
1. concentration of most camera-sampling variance in a very small Gaussian set;
2. low effective rank of per-Gaussian camera-conditioned gradient matrices;
3. prediction of cross-view gradient conflict from native alpha-compositing
   occlusion-role changes before a second backward.

The current audit did not identify a paper making all three claims for ordinary
static 3DGS training. This is not proof of novelty and must be re-audited before
submission.
