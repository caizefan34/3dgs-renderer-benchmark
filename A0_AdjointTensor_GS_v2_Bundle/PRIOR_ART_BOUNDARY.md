# A0-v2 Prior-Art Boundary

Frozen date: 2026-09-29

Known neighboring work:

- Efficient Differentiable Hardware Rasterization for 3DGS (2025):
  accelerates backward using programmable blending and hierarchical gradient
  reduction in hardware rasterization. It is a renderer/kernel redesign, not
  a coefficient-tensor basis-moment formulation.

- Faster-GS / related optimized software rasterizers:
  use faster accumulation, per-Gaussian organization, fusion, and other kernel
  engineering. They do not establish that the exact compositing backward
  coefficient field is low rank in a spatial basis.

- 3DGS-LM / matrix-free second-order methods:
  cache or apply rendering Jacobians/JVPs for solver acceleration. They do not
  replace the native per-hit alpha-compositing VJP with a tile-local coefficient
  tensor compression.

- General differentiable / adjoint rendering:
  the adjoint principle is prior art. A0-v2 does NOT claim "adjoint rendering"
  as novel.

Internal negative boundary:
the project's prior full-resolution-adjoint audit showed that knowing only the
exact full-resolution image adjoint is insufficient to skip native accepted
interactions, because alpha-compositing backward depends on contribution-specific
transmittance and accumulated/suffix-color state.

Potential novelty window if A0-v2 passes:
a 3DGS-specific factorization of the *exact contribution-level compositing
backward coefficients* into a small tile-local basis, enabling Gaussian/bucket
basis moments instead of per-pixel accepted-hit accumulation.

No novelty claim is authorized by A0-v2 alone.
