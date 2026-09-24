# F9-1R 〞 Trainable Gatherless Integration Closure

## Decision

**F9_1R_STRONG.** The inverse was removed from the production F9 path, all requested forward/backward/densification comparisons passed, and full-forward/F+B thresholds passed on every scene. The strong result authorizes〞but does not itself include〞the separate 5K training gate.

## Exact camera origin

The hot-path `torch.linalg.inv(viewmats[0])` was replaced by a one-thread-per-camera CUDA kernel that computes `-R^T t`. It is algebraically exact for `[R|t]`; observed origin error versus inverse was <=4.77e-7. Standalone median cost was about 0.0205 ms versus 0.178每0.183 ms for the inverse.

## Performance

| scene | full forward base ↙ F9-1R | gain | F+B base ↙ F9-1R | gain | derived backward delta |
|---|---:|---:|---:|---:|---:|
| room | 1.909 ↙ 1.549 ms | 18.86% | 4.209 ↙ 3.888 ms | 7.63% | +0.039 ms (+1.69%) |
| bicycle | 2.278 ↙ 1.942 ms | 14.79% | 5.541 ↙ 5.242 ms | 5.40% | +0.038 ms (+1.16%) |
| garden | 1.532 ↙ 1.178 ms | 23.15% | 2.997 ↙ 2.681 ms | 10.56% | +0.038 ms (+2.62%) |

F9-0's isolated projected-chain gain was 76.71每87.45%. In the matched F9-1R harness, old inverse-path full-forward gains were 7.29每11.96%; the camera kernel lifted them to 14.79每23.15%.

## Correctness

All scenes had identical visible IDs, radii support, intersection count, tile offsets, flatten/intersection IDs, and last IDs. RGB/alpha matched; degree-3 SH color differences were <=2.98e-7. Two H2-BWD-2R-style identical-upstream-gradient rounds against frozen scalar adjoint had zero support disagreements. F9-vs-baseline numerical deltas were comparable to the baseline replay envelope. Densification radii, proxy-gradient support, clone candidates, prune candidates, and resulting N_GS trajectory were identical; this B2 trainer has no separate split operation.

Projection VJP was 96 registers in both frozen scalar-adjoint and direct-master variants (no spills), so no cleanup was warranted.

5K was not run in this closure artifact. It is authorized by F9_1R_STRONG, but F9 cannot be frozen as SUCCESSFUL_EXACT_FORWARD_MODULE until the 5K quality/trajectory gate passes.
