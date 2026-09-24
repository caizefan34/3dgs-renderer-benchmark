# P5: external systems — SYSTEM_LEVEL_COMPARISON

**Status: DATA COMPLETE (26/26 runs per system, all rc=0).** Frozen evidence:
`aggregates/p5_external.json`, `tables/table6_p5_external.md`,
`strong_baseline_results/*/all_metrics.json` (source trees on mx).

## Scope and honest labeling

These are **system-level comparisons, not matched-protocol comparisons**. Each external
system couples renderer changes with densification/optimizer/training-loop changes, so
its wall/quality cannot be attributed to the renderer alone:

| system | commit | classification |
|---|---|---|
| faster-gs | `3cb0b755` | SYSTEM — custom CUDA backend + custom densification/optimizer |
| fastgs | `d4d33b6f` | SYSTEM — pruning-based budgeted densification + custom renderer |
| speedy-splat | `b9dd42d1` (+ submodules `b38abb6d` diff-gaussian-rasterization, `44f76429` simple-knn, `14199886` SIBR_viewers) | SYSTEM — render-speed optimization + custom training loop |

Two protocol variants per system:
- **native** — their official supported training configuration, verbatim.
- **c42** — matched camera sequence + matched eval camera set; their training config
  unchanged. This is the fairer of the two for render-side comparisons.

Per the plan's wrapper policy, the matched protocol was **not** forced onto external
systems where it would disable their mechanism (e.g. their iteration counts and
densification budgets are part of the method).

## Headline comparison (c42 variant, 13 scenes)

| system | geomean wall (c42) | mean PSNR (c42) | N range |
|---|---|---|---|
| fastgs | 3.6 min | 25.83 dB | 132K – 716K |
| speedy-splat | 12.2 min (11/13 scenes with wall) | 25.61 dB | 81K – 512K |
| faster-gs | 5.8 min | 24.61 dB | 767K – 3.29M |
| *matched B1A (reference)* | *20.8 min* | *28.57 dB* | *484K – 3.84M* |
| *matched C0 (reference)* | *19.4 min* | *28.64 dB* | *478K – 3.66M* |

The external systems trade quality for speed under their own protocols — that is their
design point, and the table records it without implying a matched comparison.

## Scene-dependent failures (reported, not hidden)

**All three external systems collapse on garden** under their own configurations:
faster-gs 13.4/13.9 dB (native/c42), fastgs 13.13 dB (c42), speedy-splat 12.37 dB (c42)
— versus ~24.2 dB for the matched arms. **Data-provenance caveat (required):** the C42
external cohort's garden runs use manually-created COLMAP data and 0.5×-area-downsampled
DSSIM (per the evidence-inventory §4 cohort table) — not the matched benchmark's garden
scene. The collapse is therefore evidence about each system's pipeline **on their garden
data variant**, not a controlled numerics failure on the standard scene; it is never
used as a matched-benchmark contrast. Within the shared failure mode, the pattern
remains noteworthy and is consistent with (but does not prove) garden-class content
stressing approximate pipelines.

Data-quality notes (recorded, not silently dropped):
- speedy-splat `bicycle`/`kitchen` wall entries are 0.0 in the ledger (wall missing;
  PSNR/SSIM/N present) — timing geomean excludes them and is labeled accordingly.
- faster-gs garden native wall (10.0 min) exceeds its c42 wall (6.7 min): its own
  protocol over-densifies garden before failing to converge.

## Provenance

- Run in the earlier strong-baselines phase on the same machine, **not** under the
  publication clean-GPU scheduler: wall times are SYSTEM_LEVEL timing and disclosed as
  such. PSNR/SSIM/LPIPS/N are eval-side measurements unaffected by GPU sharing.
- fastgs worktree dirt = untracked `__pycache__` only (commit pinning sound);
  speedy-splat's submodule pointers differ from the superproject record — the ACTUAL
  submodule commits used are pinned above.
- The 3-scene gate required by the plan is exceeded: all 13 benchmark scenes ran for
  each system.

## Use in the publication

Table 6 carries these numbers with the SYSTEM_LEVEL_COMPARISON label. The text states:
the matched benchmark (B0/B1/B1A/C0 under one trainer/protocol) is where speed and
quality claims are made; the external table situates the method among speed-focused
systems that accept large quality deltas under their own protocols, with garden as the
shared failure mode that exact-accumulation arms do not exhibit.
