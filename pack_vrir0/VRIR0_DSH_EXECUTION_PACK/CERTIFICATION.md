# VRIR0 Mathematical Certification

## A. Baseline discrete structure
For image width W, height H and tile size T, let tile grid dimensions be

Tw = ceil(W/T), Th = ceil(H/T).

For Gaussian i projected into a fixed camera at iteration t, let
m_i(t) = (u_i,v_i) and r_i(t) = (r_xi,r_yi).

For the frozen AABB path, the tile bounds are

b_i^-(t) = clamp(floor((m_i(t)-r_i(t))/T), [0,0], [Tw,Th])
b_i^+(t) = clamp(ceil ((m_i(t)+r_i(t))/T), [0,0], [Tw,Th]).

Gaussian i contributes no tile records if either radius component is non-positive. Otherwise its membership is the rectangle

B_i(t) = { (x,y): b^-_x <= x < b^+_x, b^-_y <= y < b^+_y }.

IMPORTANT: DSH must source-audit the installed gsplat 1.5.3 fork and prove that `accutile=False` selects this AABB semantics. If not, VRIR0-AABB certification is invalid and the run must stop.

## B. C-MEM: membership certificate
Given a cached visit t0 of the same camera and current visit t, accept membership only if, for every Gaussian identity i:

1. topology epoch is unchanged;
2. active_t(i) == active_t0(i);
3. b^-_i(t) == b^-_i(t0);
4. b^+_i(t) == b^+_i(t0).

Under the frozen AABB semantics this is sufficient for exact equality of the emitted Gaussian→tile multiset.

A margin form may be recorded as a diagnostic but is NOT the authoritative v0 certificate. Direct current-bound equality is clearer and less error-prone.

## C. C-ORD: order certificate
Let the cached tile list be

L_tau(t0) = (i_1,...,i_m).

At the current visit compute each Gaussian's current float32 depth sort key exactly as the baseline does. For positive finite depth, IEEE-754 float32 bit patterns are monotone with numeric depth; the gsplat reference packs those 32 bits into the low part of the intersection key.

The current fresh sorted list equals the cached list if:
- membership is unchanged; and
- the cached list is nondecreasing under the **current baseline total ordering**.

For every adjacent cached pair (i_j,i_{j+1}), require lexicographic

(depth_bits_t(i_j), emit_rank(i_j)) <= (depth_bits_t(i_{j+1}), emit_rank(i_{j+1})).

`emit_rank` is required for equal-depth ties because a stable radix sort preserves source emission order. In the frozen single-image dense path, DSH must verify source emission rank maps monotonically to the dense Gaussian/flatten index. If that verification fails, tie cases must fail closed rather than guessing.

Checking all adjacent pairs is sufficient: transitivity proves the entire cached sequence is sorted under the current total order.

## D. C-ORD-B: O(N) cached-slack certificate (implementation-oriented sufficient condition)
The exact adjacent scan above is the oracle/reference certificate, but scanning every cached intersection is O(M) in the number of intersections and may erase much of the benefit. VRIR0 therefore also measures a stronger, cheaper sufficient certificate whose revisit cost is O(N) after projection.

At cache creation, let k_i^0 be Gaussian i's positive-float32 depth key. Traverse the freshly sorted tile lists once and, for every strict adjacent pair i before j with key gap

G_ij = k_j^0 - k_i^0 >= 1,

define a symmetric integer key budget

b_ij = floor((G_ij - 1)/2).

For each Gaussian, cache

b_i = min of b_ij over all incident adjacent pairs in all tiles.

On a revisit, compute current key k_i and delta_i = |k_i-k_i^0|. If every relevant Gaussian satisfies

delta_i <= b_i,

then for every cached strict adjacent pair the current order remains strict, because

(k_j^0-delta_j) - (k_i^0+delta_i) >= G_ij - b_j - b_i >= 1.

Thus all cached tile lists remain depth-sorted. This certificate needs only current per-Gaussian depths and cached per-Gaussian `(k_i^0,b_i)`, not an O(M) revisit scan.

Equal-key pairs (G_ij=0) are not covered by the strict-gap theorem. VRIR0-v0 must either: (a) reject the view when the cached structure contains any unresolved equal-key adjacency; or (b) explicitly check only those cached tie constraints with verified stable emission-rank semantics. It must never treat a tie as a positive slack.

Record BOTH:
- `C_ORD_EXACT_SCAN`: exact current adjacency scan over cached lists (oracle/reference, O(M));
- `C_ORD_SLACK_N`: per-Gaussian key-budget sufficient certificate (future fast-path candidate, O(N)).

The implementation gate uses `C_ORD_SLACK_N` hit rate. `C_ORD_EXACT_SCAN` is used to diagnose why the stronger certificate rejects.

## E. Full certificate
CERT(v,t;t0) = C0 & C1 & C2 & C3 & C4, where:

C0 CONFIG: same camera identity/intrinsics/extrinsics/resolution, renderer semantic signature, tile size, packed mode, and supported intersection algorithm.
C1 TOPOLOGY: same topology epoch and identity mapping.
C2 NUMERIC: all fields required by the certificate are finite and supported; no invalid key/tie state.
C3 MEMBERSHIP: C-MEM passes globally.
C4 ORDER: for oracle truth diagnostics, C-ORD exact scan passes; for future implementation eligibility, the O(N) C-ORD-B slack certificate must pass (with explicit tie handling).

Any unknown state returns REJECT, never ACCEPT.

## F. Oracle truth and false-accept definition
At current visit t, the unmodified baseline produces fresh structure S_fresh(t). Let S_cache(t0) be the stored old structure.

truth_same = exact_equal(flatten_ids_fresh(t), flatten_ids_cache(t0))
             AND exact_equal(isect_offsets_fresh(t), isect_offsets_cache(t0)).

false_accept = CERT==ACCEPT AND truth_same==False.

Frozen correctness requirement: total false_accept count == 0.

False rejects are acceptable in VRIR0 because the design is intentionally fail-closed.

## G. Exactness claim boundary
A passing certificate establishes exact equality of the discrete traversal structure only. End-to-end numerical equality of a future reuse implementation still requires a later implementation-stage test showing that the renderer receives current continuous attributes and the certified cached structure without any other behavioral change.
