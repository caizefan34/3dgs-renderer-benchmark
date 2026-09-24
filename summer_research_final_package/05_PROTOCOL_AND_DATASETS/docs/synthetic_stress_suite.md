# Synthetic Stress Suite

The suite exists to expose renderer bottlenecks and scaling behavior. It does
not contain GT photographs, and its results must not be used to claim quality
equivalence.

The versioned workload catalog is
[`data/scenes/synthetic_stress_suite.json`](../data/scenes/synthetic_stress_suite.json).
Its entries are design targets for future instrumented generation and runs:

| Scene id | Class | Gaussians | Difficulty Score | Intended stress |
|---|---|---:|---:|---|
| `stress_low_overlap_50k` | Low Overlap | 50,000 | 0.8359 | Launch and scheduling overhead |
| `stress_medium_overlap_200k` | Medium Overlap | 200,000 | 2.5000 | Scaling and moderate tile density |
| `stress_heavy_overlap_400k` | Heavy Overlap | 400,000 | 5.5334 | Tile lists, overdraw, memory |
| `stress_pathological_800k` | Pathological | 800,000 | 9.4574 | Saturated overlap and depth complexity |

Scores use `geometric_mean_v1` and the four explicit inputs in the catalog.
They are workload-design scores, not measurements retroactively assigned to
historical result files.

## Historical generated scenes

The existing 50K, 200K, and 400K scenes all use the legacy clustered generator.
Observed behavior supports the labels low-density scheduling regime,
intermediate tile-list stress, and heavy/pathological tails respectively, but
the old files did not record the four difficulty measurements. Their historic
JSON remains unchanged and should show a missing difficulty score rather than
one inferred from Gaussian count alone.

## Measurement requirements

Record camera-trajectory aggregates for:

- visible Gaussian count;
- projected overlap ratio;
- average Gaussian assignments per tile;
- depth complexity over occupied image regions.

Use the same instrumentation and camera path for all renderer rows. Store raw
inputs next to the normalized score so the result remains auditable.
