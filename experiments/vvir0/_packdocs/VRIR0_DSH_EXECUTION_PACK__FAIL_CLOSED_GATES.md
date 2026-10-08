# VRIR0 Fail-Closed Gates and Verdict Policy

## Gate G0 — provenance / isolation
PASS only if:
- repository commit, environment, gsplat build/source hash, scene, seed and camera sequence are recorded;
- baseline is frozen and AABS0/FGS-H0 changes are absent from the VRIR0 worktree;
- no CUDA/C++ renderer source is changed;
- instrumentation diff is allowlisted and can be disabled to reproduce baseline behavior.

Otherwise: INVALID RUN.

## Gate G1 — semantics source audit
PASS only if installed gsplat behavior for the tested path is source-verified:
- single image;
- packed=False;
- tile_size=16;
- `accutile=False` has exact AABB membership semantics used in CERTIFICATION.md;
- raster consumes `isect_offsets` + `flatten_ids` as assumed;
- depth-key and tie/stability behavior are documented.

Otherwise: BLOCKED; do not collect evidence under an unproven certificate model.

## Gate G2 — topology
A cache entry is eligible only when cached topology epoch == current topology epoch and identity/order mapping is known unchanged. Any uncertainty rejects reuse.

VRIR0 primary verdict data are iterations 15001–30000. Earlier iterations are diagnostic, not required to pass.

## Gate G3 — numeric sanity
Reject on NaN/Inf, non-supported depth-key state, malformed radii, shape mismatch, unsupported camera/model mode, or inability to establish tie order.

## Gate G4 — membership certificate
Global v0 acceptance requires exact active-mask and cached/current tile-bounds equality for every Gaussian identity. One mismatch rejects the whole view.

Record tile-local unchanged statistics separately for a possible future VRIR1, but do not silently downgrade VRIR0 to partial reuse.

## Gate G5 — order certificate
Two order certificates are reported separately:
- `C_ORD_EXACT_SCAN`: every cached tile list is checked against current exact depth-key + tie semantics. This is the oracle/reference soundness check.
- `C_ORD_SLACK_N`: the cached per-Gaussian key-budget theorem from `CERTIFICATION.md`. This is the implementation-oriented O(N) sufficient certificate.

One violating/unknown adjacent pair rejects `C_ORD_EXACT_SCAN`; one Gaussian exceeding its certified budget (or an unresolved equal-key constraint) rejects `C_ORD_SLACK_N`.

KEEP_FOR_MINIMAL_IMPLEMENTATION requires the hit-rate gate to be satisfied by `C_ORD_SLACK_N`, not only by the O(M) exact scan.

## Gate G6 — oracle soundness
Across all required scenes and tracked revisits:
- false_accept == 0 (hard requirement);
- every certificate accept must be checked against freshly rebuilt `flatten_ids` and `isect_offsets`.

Any false accept => FAIL CLOSED. The certificate implementation must be fixed and the affected evidence rerun from scratch.

## Gate G7 — opportunity / Amdahl screen
Measure or import baseline time for the *actually removable* structure work:
T_struct = T_intersection + T_sort + T_offset.

For certified hit rate h, the optimistic zero-cost-certificate upper bound is

S_UB = T_iter / (T_iter - h*T_struct).

This is deliberately optimistic: it assumes perfect reuse costs zero and ignores cache/certificate overhead.

Implementation is forbidden if S_UB < 1.05 for the target canonical training regime even at the observed h. If S_UB < 1.05 even for h=1, DROP implementation regardless of structural coherence.

Report renderer-only and true full-iteration bounds separately; never transfer one to the other.

## Gate G8 — hit-rate screen
For global-view VRIR0 to advance:
- post-15k full-view certified hit rate >= 0.60 on each required scene; and
- G7 must pass on each target regime.

If full-view hit rate fails but tile-local structural equality >= 0.90 on each scene, verdict is `REDESIGN_TO_TILE_GRANULAR_VRIR1`, not KEEP.

These are project decision thresholds, not theoretical constants.

## Gate G9 — cache footprint
Future implementation projection:
- additional GPU-resident cache budget <= 10% of baseline peak VRAM;
- all-view cache estimates must use measured `flatten_ids/isect_offsets` sizes, not Gaussian count alone.

Oracle data may be CPU-offloaded and limited to tracked cameras. If projected cache exceeds budget, global all-view implementation is not permitted; evaluate bounded LRU/host metadata only after VRIR0.

## Required multi-scene evidence
Primary: room, bicycle, garden.
At minimum report late-training windows:
- 15001–20000
- 20001–25000
- 25001–30000

For every scene/window report:
- same-view revisit count;
- topology-eligible revisit count;
- full-view truth-same rate;
- C-MEM pass rate;
- C-ORD exact-scan pass rate conditioned on C-MEM;
- C-ORD O(N) slack-certificate pass rate;
- full fast-certificate pass rate;
- false accepts / false rejects;
- tile-local unchanged fraction;
- adjacent order violations per million pairs;
- n_isects and cache bytes;
- cache age in iterations;
- projected drift p50/p95/p99 for means2d/radii/depth;
- optimistic S_UB.

## Frozen verdict mapping
KEEP_FOR_MINIMAL_IMPLEMENTATION only if G0–G9 all pass.
REDESIGN_TO_TILE_GRANULAR_VRIR1 only if soundness is perfect, global hit rate fails, and tile-local coherence passes G8.
DROP if opportunity/Amdahl or cache economics fail despite coherence.
BLOCKED if semantics/provenance are not auditable.
