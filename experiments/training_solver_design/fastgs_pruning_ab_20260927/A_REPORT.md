# Room@10K static opacity A gate

**Decision: stop.** The preregistered A gate required at least 16 high-cost candidates. Only 13 of the top-64 cost candidates qualified, so no candidate set was frozen, no B image was rendered, and no control or garden experiment was run.

The experiment used the corrected author FastGS room@10K `eval=True`, p=1 PLY at 1600×1066. Its SHA-256 was `4f0b7032dd97da9da1807a2047ff12a4d8c871ffe7e465aa212397aedae9cff3` both before and after. The source was commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`; `train.py` matched its preregistered SHA-256. Only in-memory raw-opacity edits were made, with each value restored and checked. No PLY or checkpoint was written.

## Preregistered gate and outcome

The eight A training views were fixed by PCG64 seed `20260927`: `DSCF4687`, `DSCF4758`, `DSCF4792`, `DSCF4903`, `DSCF4928`, `DSCF4940`, `DSCF4951`, and `DSCF4972`. The eight B names were also fixed in the preregistration, but B was not loaded or rendered because A failed. The 39 held-out/development cameras and the post-hoc `DSCF4771` diagnostic were not used.

No-edit baselines were rendered three times per A camera. The largest repeat deviation from a camera median was exactly 0, so the frozen threshold was `epsilon = max(1e-6, 5 × noise) = 1e-6`. A candidate qualified only if its mean eight-view original-loss delta was at most `-1e-6` and at least one A view improved by at least `1e-5`.

Of 306,048 Gaussians visible in at least two A views, the top 64 by summed all-ones accepted-pixel count were tested individually on all eight A views. Thirteen qualified, below the required 16. The largest-cost qualifying candidate was PLY row `60448`; the complete 13-row list is below.

| PLY row | A accepted-pixel count | Mean A loss change | A views with change ≤ −1e−5 |
|---:|---:|---:|---:|
| 60448 | 615,745 | −1.2537278235e−5 | 4 |
| 634778 | 331,436 | −8.3093531430e−6 | 2 |
| 617483 | 203,961 | −2.8331764042e−6 | 1 |
| 625427 | 200,965 | −1.2325122953e−6 | 1 |
| 548765 | 197,286 | −1.9015744329e−6 | 1 |
| 621238 | 181,330 | −2.0068138838e−6 | 1 |
| 570522 | 180,673 | −1.5858560801e−6 | 1 |
| 619360 | 166,533 | −3.8413330913e−6 | 1 |
| 360280 | 165,025 | −1.8198043108e−6 | 1 |
| 681694 | 153,187 | −2.3152679205e−6 | 1 |
| 608665 | 140,941 | −4.3075531721e−6 | 2 |
| 266491 | 135,187 | −3.4500844777e−6 | 1 |
| 111543 | 134,926 | −5.3665600717e−6 | 2 |

## Work-count bound and cost

Across the eight A views, the renderer counted `672,708,370` accepted Gaussian–pixel contributions over the full model. The 13 qualifying rows' summed individual counts are `2,807,195`, or **0.4173%** of that total. The highest-cost 16 rows in the pool sum to `5,025,977` (**0.7471%**); the entire top-64 pool sums to `12,755,820` (**1.8962%**).

These are additive, optimistic work bounds from individual baseline counts. They are not measured group savings: after opacity suppression, overlapping points may not translate into the same net count reduction. The experiment did not run a group counter because the A gate failed. The API reports accepted Gaussian–pixel contributions, not per-Gaussian tile instances, and these counts do not imply training speed.

The A stage took 3.072 seconds of synchronized stage wall time on physical GPU 2. It used 24 no-edit baseline renders, 8 all-ones counter renders, and 512 single-point intervention renders. The room mechanism failed before any B validation; the expected B quality metrics, adaptive FastGS control, and full training speed/quality remain unmeasured.

## Reproduction and integrity

The frozen pre-registration SHA-256 is `c542916a15f5cc13748e31dd4c069b1dbe64d6496a0518725abef3fb6be408cd`; the A script SHA-256 is `ff30295992a32ad2eabbedb344dd28e1492aa1c406a32ce2ece250d811242c88`. The raw result SHA-256 values are:

- `a_result.json`: `c3721e35327c2acab699d106572824d9714c439c0e66f64e8cd99da582c56705`
- `candidate_scan.jsonl`: `b0f1d9c459440fbc47d28d249bd6c6a3325ae8e5ed7eee7e4a1ab55ae3dc28c5`
- `a_counts_radii.npz`: `238c9094b45ac4ed980b2b9e4494a72774913ddb197c2d1bfa82752d6dabf23a`

Raw outputs are in `results/a_stage/`. One launch failed at a pre-render assertion because B cameras were intentionally not constructed; it produced no render or metric. The assertion was corrected, the preregistration was amended before A measurement, and the final script hash above is the corrected script hash.
