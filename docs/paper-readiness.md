# Paper-readiness audit

Status date: 2026-08-06. Target assumed here: a graphics/systems benchmark or
artifact paper. A HiGS algorithm paper needs a separate novelty and ablation
argument.

## Executive assessment

The repository already has unusually strong artifact foundations: immutable
asset and protocol hashes, raw repeat samples, evidence-tier separation,
quality gates, complete-suite ranking rules, CPU-safe tests, and a public
dashboard. The current evidence supports claims about the fixed five-scene
A100 cohort. It does not yet support a universal renderer ranking or a broad
hardware-generalization claim.

The shortest credible path is to publish the benchmark as the primary
contribution and present compression as one bounded case study. HiGS training
should remain a separate paper because combining a benchmark, compression
survey, native backward implementation, and training algorithm as co-equal
contributions makes the paper harder to evaluate and weakens the independence
story.

## Implemented evidence gates

- `paper/claims.json` binds supported claims to Git-tracked JSON evidence with
  SHA-256 hashes and executable assertions. Current status is 4 supported,
  3 blocked, and 1 out of scope.
- CI checks the claims manifest and all public relative links with tracked-file
  enforcement.
- A deterministic, self-verifying evidence bundle is built and attached by the
  `v*` tag release workflow. A real archival DOI still requires publishing the
  tagged release to an archive such as Zenodo.
- The README now limits headline results to the measured A100 cohort, provides
  direct evidence and result-submission paths, and removes duplicated research
  history without removing the underlying tracked reports.

## Submission gates

| Priority | Gate | Current evidence | Acceptance criterion |
| --- | --- | --- | --- |
| P0 | Freeze one contribution narrative | README mixes benchmark, compression, and HiGS training claims | One abstract-level question, at most three contributions, and a claim-to-figure map |
| P0 | Independent replication | Tier A headline uses one A100 host | At least two additional GPU architectures, including one consumer GPU, with separate cohorts and one run by a second operator |
| P0 | Experimental independence | Five repeats reuse one initialized process and camera sequence | At least three fresh-process blocks per renderer/case; randomize or balance order and report block/day effects |
| P0 | Statistical contract | Per-case Student-t intervals exist; aggregate bounds are geometric means of marginal bounds | Report repeat-level CV, interval method, effect size, and a hierarchical or blocked analysis for aggregate claims |
| P0 | Baseline coverage | Five families have complete Tier A coverage; several registry entries lack adapters | Add the strongest recent exact and approximate renderers, or justify exclusions with a dated systematic search protocol |
| P0 | Archival release | GitHub source and raw artifacts exist; no repository DOI is declared | Tagged release, immutable archive (for example Zenodo), DOI in `CITATION.cff`, checksummed artifact bundle, and frozen paper tables |
| P1 | Dataset breadth | Five scenes from two datasets | Add indoor/outdoor and low/high Gaussian-count coverage from at least one additional benchmark dataset, then run sensitivity analysis |
| P1 | Temporal quality | Static PSNR/SSIM/LPIPS are complete | Add camera-path flicker/popping metrics and visual sequences for approximate or reordering renderers |
| P1 | Resource control | Environment and power fields exist, but the headline cohort is not cross-host | Lock/report clocks and power, record temperature, and repeat unstable TC-GS runs after steady-state warm-up |
| P1 | External validity | No public external result submission has been accepted | Accept and audit one independent cohort through the result-submission template |
| P1 | Artifact evaluation | Setup scripts, containers, and tests exist | Clean-machine reproduction under a documented time/storage budget with a one-command smoke path |
| P2 | Governance | MIT, contribution guide, issue templates, and discussions exist | Named maintainers, review policy, release cadence, changelog, and published benchmark-version policy |

P0 items are paper blockers. P1 items are likely reviewer concerns; a paper may
proceed only if the missing item is explicitly scoped out and claims are
narrowed. P2 items mainly affect long-term authority and adoption.

## Statistical design for the next matrix

Use a randomized blocked experiment. A block is one fresh process execution of
all renderer/case combinations on one host under a recorded clock and power
policy. Rotate renderer order inside each case, retain repeat and frame
structure, and never count the 100 correlated camera frames as 100 independent
replicates.

Primary analysis should report per-case speed ratio against Original 3DGS and
the geometric mean ratio across the fixed case set. Bootstrap blocks/repeats
inside each case for uncertainty; treat scenes as fixed benchmark tasks unless
the paper explicitly defines a population of scenes. Report raw ratios and
confidence intervals alongside practical quality tolerances. Do not infer a
win merely from non-overlapping marginal intervals.

The result collector now records pooled FPS, repeat-mean FPS, repeat median,
repeat CV, repeat count, and the interval method for new runs.
Historical artifacts remain immutable and can derive these values from their
hashed repeat arrays.

## Claim boundaries

Safe claim: "On the frozen five-scene A100 cohort, renderer X achieved Yx the
speed index of Original 3DGS under protocol Z while remaining within the stated
quality tolerances."

Unsafe claim: "X is the fastest 3DGS renderer" without a cohort qualifier.
Also avoid "measured them all": the registry itself documents unsupported and
not-yet-integrated renderers.

Benchmark and method independence must be explicit. If HiGS code is authored
or modified in this repository, the evaluation should name that conflict,
freeze configurations before the final run, and include externally maintained
baselines under the same protocol.

## Authority and adoption plan

1. Cut a small, reproducible `v0.3.0` release after the P0 protocol changes;
   archive it and add the issued DOI rather than inventing one in advance.
2. Publish one stable paper table and one interactive explorer backed by the
   same machine-readable artifact; add a generated-data consistency check to CI.
3. Invite renderer authors to verify adapters and submit signed-off corrections
   without granting any project a special ranking path.
4. Announce reproducible releases with a concrete result, hardware scope, and
   artifact link. Stars are a consequence of utility and trust, not a scientific
   metric and not evidence of authority.
5. Maintain a dated coverage table showing integrated, blocked, and excluded
   methods with reasons. This is more credible than maximizing a renderer count.

## Recommended paper structure

1. Problem: cross-paper renderer numbers are not comparable.
2. Protocol: immutable workloads, cohort identity, timing, quality, and failure policy.
3. Artifact: adapters, provenance, validation, and contribution workflow.
4. Evaluation: cross-renderer, cross-scene, cross-hardware results with uncertainty.
5. Case study: one bounded compression analysis, not a second paper inside the paper.
6. Threats to validity: host count, dataset selection, implementation ownership, temporal metrics, and upstream-version drift.
