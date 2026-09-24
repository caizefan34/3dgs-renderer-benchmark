# F9-0 - Gatherless Projected Primitive Producer

## Verdict

**F9_STRONG.** Exact-support and F4 structural gates pass on all three 2048-max-side cam0 fixtures. The forward-only prototype is authorized for a narrowly scoped trainable integration that preserves B2 backward-required state; this patch does not modify backward, F4, or F5.

## Timing

| Scene | F1 | F2 | F3 | F1+F2+F3 | F9 | Front-end gain | Full forward gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| room | 0.068 | 0.097 | 0.347 | 0.482 | 0.065 | 86.62% | 26.12% |
| bicycle | 0.144 | 0.097 | 0.347 | 0.479 | 0.112 | 76.71% | 19.47% |
| garden | 0.056 | 0.096 | 0.346 | 0.481 | 0.060 | 87.45% | 33.65% |

All values are CUDA-event medians from 20 warmups plus 5x100 interleaved samples. F+B has no F9 value because this is explicitly a forward-only producer and no backward contract was changed.

## Exactness and resources

The implementation preserves FP32, SH degree 3, pinhole projection, clamp, visibility, and F4/F5. It is **ALGEBRAIC_EXACT_FP_REASSOCIATED** rather than bitwise order-exact: means2d max absolute difference is <= 9.77e-4 and conic <= 4.66e-4, while colors/depths/opacities are exact, support mismatch is zero, and F4 intersections/offsets/flatten IDs/non-tie ordering are identical.

F9 uses 48 registers/thread, zero shared/local memory, and zero spills (62.5% theoretical occupancy). Baseline FP32 projection uses 40 registers (75%); SH3 uses 62 (50%). The gain comes from replacing gather/projection/direction/SH/clamp launches and eliminating 260 B/G temporary writes plus 276 B/G rereads logically. No DRAM counter claim is made.

## F9-B

Do not test descriptor hoisting: the frozen F4 computes bbox and pair emission in one kernel, so no COUNT/FILL duplicate predicate exists to amortize a 16 B/G descriptor.
