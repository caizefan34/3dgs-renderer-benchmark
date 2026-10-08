# MomentGrad-GS — Stage M1 Report (Quadrature Oracle)

Date: 2026-10-03
Lane: MOMENTGRAD-GS (compressed adjoint integration for Gaussian Splatting backprop)
Hardware: RTX 5070 Laptop (8GB), torch 2.13+cu130, Python 3.13. NOT A100.
Backend note: gsplat 1.4.0 fused CUDA backend cannot JIT-compile on this host (no MSVC
`cl` compiler). All M0-M4 work uses a self-contained pure-PyTorch oracle rasterizer built
for this lane (`momentgrad_gs/oracle.py`). M6 (CUDA prototype) and M7 (A100 authoritative)
are BLOCKED on this hardware and are NOT fabricated.

## Method
For each Gaussian the first-order backward field is
    H(p) = dL/dI(p) . C_i . T_i(p)
and the gradient terms are Gaussian-weighted moments of H over the screen footprint:
    g_mean2d = Sigma^{-1} sum_p H(p) alpha(p) delta_p
    g_opacity = (1-opa) sum_p H(p) alpha(p)
    g_color  = sum_p dL/dI(p) T_i(p) alpha(p)
We approximate the footprint sum with K-point tensor-product Gauss-Hermite quadrature
against the screen Gaussian (exact for polynomial degree <= 2n-1, K=n^2), evaluating H
(incl. exact front transmittance T_i at the K sample points) and compare to the canonical
footprint integral (oracle moments, exact T at footprint pixels).

Oracle validation (single Gaussian): alpha-peak == opacity (0.0999 vs 0.100); quadrature
matches canonical pixel-sum gradient to 0.4% (ratio 1.004). Normalization factor 2*pi*det(L)
verified.

## Results (room/garden/bicycle, 500 on-screen Gaussians, sigma~15px render)
Median per-Gaussian cosine (typical case):
    opacity  ~1.0000    color 0.997-1.000    geometry(mean2d) 0.84-0.98
Global/aggregate cosine (all Gaussians concatenated):
    opacity 0.79-0.98   color 0.88-0.99      geometry(mean2d) 0.74-0.95
Work reduction (interactions / (K*visible)):
    550x (K=16) to 2200x (K=4)
K-sweep non-monotonic: K=9 frequently worse than K=4 (e.g. room opacity gcos K4=0.861,
K9=0.792, K16=0.975).

### Smooth-field control (isolates the cause)
With a smooth low-frequency adjoint field, quadrature is near-perfect AND monotonic:
    K=4  mean2d gcos=0.99505  opa=0.99968  color=0.99975
    K=9  mean2d gcos=0.99701  opa=0.99996  color=0.99997
    K=16 mean2d gcos=0.99817  opa=0.99999  color=0.99999
=> The real-field failure is caused by NON-SMOOTHNESS of the alpha-compositing adjoint
field (occlusion boundaries, tile edges, neighbor discontinuity), not by the quadrature.

## OBSERVED
- Typical Gaussian opacity/color backward is compressible to ~1.0 cosine with K<=16 at
  >500x work reduction.
- Aggregate (global) gradient cosine is below the 0.995 gate on all three scenes.
- Geometry (mean2d) aggregate is the hardest (0.74-0.95), below the 0.99 gate.
- Huge-footprint Gaussians at occlusion boundaries dominate the aggregate error
  (canonical |g| up to ~2e5).

## EVIDENCE
- `momentgrad_gs/out/m1_quadrature.json` (per-config metrics).
- `momentgrad_gs/out/M1_SMOOTH_CONTROL.json` (control).
- Single-Gaussian validation: quadrature/canonical ratio 1.004.

## INFERENCE
- The compositing adjoint field is non-smooth exactly where footprints are large, so
  low-order quadrature does not converge monotonically and the aggregate gate fails.
- This is the core difficulty a depth-aware moment representation (M2/M3) or a difficulty
  certificate with fallback (M4) would need to address.

## HYPOTHESIS (untested here)
- A depth-aware moment basis (M2/M3) might restore smoothness per tile x depth bucket and
  recover the gate; not yet tested.

## NOT_MEASURED
- M2 moment-field oracle, M3 depth-aware compression, M4 certificate + E2E headroom,
  M5 training convergence, M6 CUDA, M7 A100.

## CONCLUSION (M1 gate)
Pre-registered gate: for K<=16 require global gradient cosine >= 0.995 AND geometry
aggregate cosine >= 0.99 AND interaction reduction >= 4x, on all three scenes.
Result: work reduction PASSES (>=550x); global and geometry cosine FAIL on all scenes.
Per the pre-registered rule: MOMENTGRAD = DROP_GRADIENT_FIDELITY (STOP; do not rescue by
raising K, since cost advantage would disappear).

Positive sub-result worth preserving: the typical-case opacity/color backward IS
compressible (>500x, ~1.0 cosine), suggesting a certificate-gated variant (compress the
easy majority, fall back on hard occlusion-boundary Gaussians) is the only viable path.
