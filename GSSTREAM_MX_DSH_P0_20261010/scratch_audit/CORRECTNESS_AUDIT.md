# Correctness audit — cpu_oracle.py / test_properties.py (independent)

Auditor note, CPU-only. Date 2026-10-10. No GPU. This is an independent review
of the shipped oracle; it does not modify the packaged files.

## What the shipped tests actually prove
- `test_properties.py`: 400 random single-Gaussian SPD/opacity/anisotropy trials;
  asserts every pixel whose recomputed alpha >= CUTOFF lies inside the conservative
  row interval. This is a genuine **no-false-negative** test of `xy_bounds`. Sound.
- `cpu_oracle.py` dense vs sparse: dense visits `rank` (depth-sorted) at every pixel;
  sparse visits per-pixel row-interval streams. Parity `max_abs_err == 0.0` on
  rgb/gcol/gopa/gmean AND `dense.hits == sparse.hits`.

## Key audit finding: the parity test is NOT independent
The dense and sparse paths share ALL of: `alpha()`, the CUTOFF, TERMINATION, the
compositing recurrence, the early-stop break, and the reverse-mode gradient block.
The ONLY thing sparse changes is which (gaussian,pixel) pairs are visited, via
`xy_bounds`. Therefore the 0.0 parity result is a **tautology over shared code**:
it can only detect a row-interval *false negative/positive* (a culling bug). It
CANNOT detect a wrong alpha formula, wrong cutoff, wrong composite, or a wrong
gradient — because an error there would appear identically in BOTH paths and cancel.

Consequence: `CPU_BASIC_PARITY_PASS` is weaker than it looks. It validates the
*interval culling*, not the *renderer math*. The kit's own limitations array
("GPU float cutoff/clamp and sort semantics not verified", "NOT native training
parity") is consistent with this.

## Independent check performed (the missing independence)
Added `scratch_audit/fd_gradcheck.py`: central finite differences of the toy
RGB-MSE forward render vs the oracle's analytic gcol/gopa/gmean. Result:
- opacity grad: max|analytic-fd| = 1.44e-9
- mean2d grad : max|analytic-fd| = 1.08e-9
- color grad  : max|analytic-fd| = 1.61e-9
So the **analytic gradients are genuinely correct** for the toy objective — this is
the correctness the parity test could not establish. (Conic gradients are not
exposed by the oracle, so they are FD-unchecked; native 3DGS backprops through the
conic via 2D covariance from 3D — not modeled here at all.)

## Boundary / edge-case review
- `alpha()` uses `exp(-max(0,sigma))`; sigma is a PSD quadratic form so >=0 always;
  clamp min(.999,.) only ever lowers alpha. Since CUTOFF=1/255 << .999, the clamp
  never causes a valid pixel to be dropped. No false negative from clamp. OK.
- Opacity grad guard `if raw<.999` correctly zeroes the opacity/mean2d gradient in
  the clamped regime (alpha constant there) — matches FD. OK.
- `xy_bounds` adds `+2`/`-2` conservatism and clamps discriminant to >=0: strictly
  conservative, never drops a supported pixel. OK. (Inflates candidate counts;
  relevant to cost model, not correctness.)
- `tau = 2*log(o/CUTOFF)` requires o>=CUTOFF; `xy_bounds` early-returns when
  o<CUTOFF, so log arg >=1, tau>=0. No NaN. OK. (o==CUTOFF exactly -> tau==0 ->
  single-pixel interval; degenerate but safe.)
- `validate()` asserts SPD (a*c-b*b>0) and 0<=o<=1. Hard assert, NOT a dense fallback
  as the task's G3 "near-singular anisotropic conics" robustness clause expects.
  This oracle cannot exercise near-singular/non-SPD conics — it aborts. This is a
  **gap against preregistered G3**, not a silent bug.

## Verdict for the correctness gates
- G1 (CPU correctness, property + parity): **PASS**, with the explicit caveat that
  parity is non-independent and native semantics (float cutoff/clamp/sort/absgrad/
  SH) are untested by design. Gradients independently FD-verified correct.
- G3 (robustness): the *property/gradient* part is sound, but the oracle lacks a
  dense-fallback path for non-SPD/near-singular conics, so the preregistered
  near-singular robustness scenario is NOT testable with this code as shipped. This
  is a HOLD-level gap for any claim that depends on production robustness, not a
  correctness failure of what is present.
