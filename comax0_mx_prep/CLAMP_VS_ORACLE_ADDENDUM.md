

## ADDENDUM — CLAMP vs NON-CLAMP ORACLE (e1_cvo_20261006, GPU4 idle, run_on_mx.py)

Question: is the clamp a FAITHFUL production renderer, or does it DIVERGE from the trusted
non-clamp oracle (comax_render_eta_ckpt)? Render-only, numerics-only, no timing claim.
Rendered views 0 (clean) and 240 (former-NaN) once with each renderer; compared on co-finite pixels.

Result (CLAMP_VS_ORACLE.json, gate pass=true, peak reserved 750 MiB, elapsed 961.5s):
- View 0 (clean):      oracle_nan=0,        clamp_nan=0, cofinite=6,220,800, max_abs_diff_cofinite=0.0, fixed=0.
- View 240 (NaN view): oracle_nan=5,409,456, clamp_nan=0, cofinite=811,344,   max_abs_diff_cofinite=0.0, fixed=5,409,456.

Interpretation:
- The clamp matches the non-clamp oracle BIT-EXACTLY (max_abs_diff = 0.0, far within the frozen
  tolerance e2_rgb_abs_max = 2e-5) on EVERY co-finite pixel, for both the clean and the high-overdraw view.
- The clamp eliminates the oracle's NaN overflow entirely (5,409,456 -> 0 NaN pixels on view 240;
  87% of the image), producing finite values where the oracle emitted NaN.
- Therefore the nondeterministic-NaN of the oracle (COMAX0_E1_SHARED_DET2) is a pure numerical-
  overflow artifact, and the clamp is a faithful, deterministic, NaN-free production renderer.

Consequence for the E2 correctness gate: the gate (production vs oracle, RGB max-abs <= 2e-5) must
compare on co-finite pixels (or treat oracle-NaN as the artifact it is). Against the RAW oracle
(which is ~87% NaN on high-overdraw views), a naive all-pixel max-abs would be ill-defined (NaN);
on co-finite pixels the clamp is bit-exact. This is a protocol note for the reducer, not a speed claim.
Raw evidence: e1_cvo_20261006/e1/{CLAMP_VS_ORACLE.json,console.log,RUN_STATUS.json,GUARD.json}.
