# H4-P0 — Exact B16 Pre-exp Support Filter

**Decision: H4_P0_WEAK — DROP H4 FAMILY.** The analytic predicate is sound, but its inlined construction costs 70 additional registers in the existing F5 specialization and makes F5 substantially slower on every target scene.

## Predicate

For each 4×4 pixel-center box, H4-P0 analytically minimizes

```text
sigma(dx,dy) = 0.5*(A*dx*dx + C*dy*dy) + B*dx*dy
```

over its axis-aligned domain. It evaluates the four corners and the clipped stationary point on every edge; a block is live when `min_sigma <= log(255*opacity)`. The continuous-box minimum is conservative with respect to the 16 discrete pixel centers. The existing `sigma < 0` branch is retained.

The standalone CUDA predicate was checked against exhaustive 16×16 alpha evaluation on 100,000 real tile-G entries per scene: false negatives were **0/300,000**. Conservative false-positive blocks were room=68, bicycle=519, garden=85. Construction used 64 registers, no stack or spills, and 0.269/0.269/0.280 ns per mask in the isolated launch (room/bicycle/garden).

## Integration and resources

The unchanged F5 256-thread batch is partitioned into eight 32-G warp subgroups. Each lane builds one `uint16` B16 mask. The warp bit-transpose produces 16 depth-ordered `uint32` block words per subgroup in transient shared memory (512 bytes per CTA); pixels test their block word before existing sigma/exp/alpha/compositing. No F4/sort/backward/compositing order change was made.

| F5 specialization (RGB, 16×16, 256 threads) | Baseline | H4-P0 |
| --- | ---: | ---: |
| registers/thread | 31 | 101 |
| dynamic shared memory | 7,168 B | 7,680 B |
| spills | 0 | 0 |
| static barriers/batch | 2 | 3 |
| estimated occupancy | 100% | 25% |

The +70-register delta is a major occupancy cliff, violating the resource gate even though no spill occurs.

## Direct F5 diagnostic timing

CUDA events, 20 warmups, five 100-sample repetitions per variant. The two variants required fresh processes due their mutually exclusive `gsplat` operator registration, so this is not interleaved; it is sufficient only as a resource-gate diagnostic.

| scene | baseline F5 (ms) | H4-P0 F5 (ms) | gain |
| --- | ---: | ---: | ---: |
| room | 0.707584 | 1.387520 | -96.09% |
| bicycle | 0.914432 | 1.823744 | -99.44% |
| garden | 0.413696 | 0.891904 | -115.66% |

The compact F5 output signatures (render sums/maxima, alpha sums/maxima, and last-id sums/maxima) match in all three scenes. A full tensor-diff check was not performed.

## Mechanism accounting

The corrected H4-0R exhaustive oracle predicts B16 skips 65,631,268 / 151,845,476 / 33,167,852 sigma and exp evaluations (room/bicycle/garden), corresponding to 36.34% / 53.79% / 30.74% of visited pairs. This promising arithmetic saving is overwhelmed by mask generation, transpose/publication, and the register-pressure cost. Because the resource gate already failed, no production in-kernel counter run, full-forward timing, or F+B timing was executed.

No novelty claim is made: opacity-aware ellipse support, subtile culling, and warp masks are prior-art categories. The pending HiGS-native integration claim is not pursued because this gate failed.

## Deliverables

`artifacts/higs-h4-p0/` contains the predicate, ptxas, timing, correctness, resource, and gate records. `patches/higs-h4-p0-b16-prefilter.patch` is the forward-only source patch.
