# AtomBirth-GS AB0 — PREREG STUB (BLOCKED, not run)

## STATUS
AB0_BLOCKED — NOT RUN. This stub records why AB0 cannot execute under the current
frozen constraints and what a future run would require. No experiment is launched.

## WHY BLOCKED
1. AB0's scientific cells require a canonical room@5K / bicycle@5K early-stage
   checkpoint as the parent-free-atom birth state.
2. The frozen C0 trajectory (n1snap, sha256 49a79538..) saved snapshots ONLY at
   10K/15K/20K/25K. NO canonical 5K exists anywhere.
3. Regenerating 5K from the frozen trainer is CROSS-RUN NON-DETERMINISTIC
   (custom CUDA rasterizer backward atomicAdd; torch determinism APIs cannot
   control it). Regenerated 10K/15K/20K/25K already MISMATCH canonical (8/8);
   a regenerated 5K is therefore NOT canonical and MUST NOT be substituted
   (user decision A: do not call regenerated checkpoints canonical; do not relax
   the bit-exact identity gate).
4. AB0's primary metric R_parent (parent-free vs parent-derived held-out
   improvement) and P_full8 both rely on held-out descent, which LAW-MATURE-HELDOUT
   shows is unreliable at mature checkpoints; AB0 must be defined on an EARLY
   state where native held-out descent is positive.

## WHAT A FUTURE UNBLOCK REQUIRES (separate prereg + sign-off)
- A NEW preregistered "stochastic-replicate early-stage" protocol that:
  * Defines its OWN analyzer and verdict namespace (AB0_*), never reuses the
    original AB0 analyzer on a reduced/substituted cell set.
  * Explicitly treats the birth checkpoint as a STOCHASTIC REPLICATE cohort
    (multiple seeds / multiple regenerated 5K), with population-level (median,
    p10) inference rather than a single bit-exact checkpoint.
  * Pre-registers the birth-state cohort size, the held-out camera discipline,
    and the gates (ATOM positive>=6/8, median R_parent>=1.5, p10>=0.75,
    median P_full8>=0.10, quality guard) BEFORE any run.
  * Does NOT substitute p13x/VOS or any non-canonical checkpoint.
- Until such a prereg is signed off, AB0 remains BLOCKED. Do NOT run.

## ARTIFACT
- This stub: 3dgs-renderer-benchmark/h0_run/AB0_PREREG_STUB.md
- Original AB0 blocker: CLOSED_DIRECTIONS_20260930.md INFRA-C0-NONDETERMINISM