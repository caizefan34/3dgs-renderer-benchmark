# AccuTile A0 opportunity gate

Result: `PASS FOR WORKLOAD OPPORTUNITY; BLOCKED BY CORRECTNESS`.

Same mature checkpoints and real camera 0 as frozen P2I. Modern upstream
AccuTile was applied to the identical projected tensors. Its AABB counts exactly
match frozen P2I counts.

| Scene | Visible | AABB isects | AccuTile isects | Reduction | Mean tiles AABB/Accu |
| --- | ---: | ---: | ---: | ---: | ---: |
| room | 16,076 | 25,475,247 | 12,334,640 | 51.58% | 1584.68 / 767.27 |
| bicycle | 13,654 | 18,550,853 | 9,171,849 | 50.56% | 1358.64 / 671.73 |
| garden | 100,582 | 11,257,252 | 6,249,412 | 44.49% | 111.92 / 62.13 |

All three scenes exceed the strong opportunity threshold. Full distributions
(p50/p90/p95/p99/max) are retained in
`experiments/final_sprint/accutile/raw/a0_results.json`.

This result is not permission to benchmark a lossy path. The required
pixel-support check failed, so A1 was not started.
