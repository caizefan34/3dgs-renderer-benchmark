# DEA-GS M0 Verdict

**VERDICT:** `DROP_INSUFFICIENT_HEADROOM`

## Binding reason

Under the pre-registered protocol, the headroom gate is binding and overriding. The measured optimizer fraction is only 5.94% of the iteration — below the 15% immediate-drop threshold — and the conservative projected end-to-end speedup is 0.973x, below the required 1.25x. Even granting the full 4.04x appearance-work reduction and a safe lifecycle, the optimizer is too small a share of the iteration for appearance-only lazy Adam to produce a net speedup; under conservative assumptions (adverse optimizer fraction 0.0624, p90 visible-fraction residual, >=2x bookkeeping overhead) the method is net-sLOWER. Independently, the numerical oracle also fails on real states (theta p99.99 rel 3.88e-06 > 1e-6, theta max rel 1.09e-03 > 1e-5, fallback 11.4% > 5%), so the closed-form catch-up is not certifiable to tolerance without excessive dense fallback. No threshold was changed, no geometry/opacity semantics were altered, no CUDA was written, and no rescue was attempted.

## Gates

| Gate | Threshold | Result | Pass? |
|---|---:|---:|---|
| Baseline profile valid | required | valid=true (real mx continuation, N=933590, 500-iter window) | yes |
| Lifecycle safe | required | safe=true (all observe/copy/remap events materializable) | yes |
| Appearance update work reduction | >= 3.0x | 4.037x | yes |
| theta rel p99.99 | <= 1e-6 | 3.88e-06 | no |
| theta rel max | <= 1e-5 | 1.09e-03 | no |
| m rel p99.99 | <= 1e-7 | 9.6e-16 | yes |
| v rel p99.99 | <= 1e-7 | 6.0e-16 | yes |
| fallback fraction | <= 5% | 11.4% | no |
| conservative projected E2E | >= 1.25x | 0.973x | no |

## Authorization

- Verdict is `DROP_INSUFFICIENT_HEADROOM` (not `PASS_TO_M1`). STOP. No rescue in this lane without a newly pre-registered hypothesis.
- Per protocol section 9, do NOT implement CUDA or run long training even as a follow-up in this lane. M0 killed the weak idea cheaply; that is the intended outcome.
