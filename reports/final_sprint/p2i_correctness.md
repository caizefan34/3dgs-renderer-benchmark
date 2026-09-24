# P2I correctness

`P2I_FUSED` was not implemented because P0 failed. Downstream ID, render,
gradient, and training checks are `NOT_RUN`, not inferred passes.

| Scene | Production vs extracted `tiles_per_gauss` mismatches | Max absolute difference |
| --- | ---: | ---: |
| room | 0 | 0 |
| bicycle | 0 | 0 |
| garden | 0 | 0 |

This is a P0 control for the measurement extraction, not a fused-path claim.
