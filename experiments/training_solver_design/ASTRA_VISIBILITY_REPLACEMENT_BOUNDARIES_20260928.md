# Visibility/representation replacement: bounded rejection notes

2026-09-28. Read-only literature work. No experiment or renderer/trainer implementation. On root instruction, the shared-surface-alpha line is stopped; current work waits for the existing correlated-direction birth Stage0 result.

## Directly occupied candidate

The concrete candidate considered was to replace repeated per-Gaussian occlusion on one reconstructed surface with a normalized surface cohort: group co-surface contributions, reconstruct their color jointly, compute one coverage opacity for the group, and composite between surfaces. The intended quality path was avoiding incorrect repeated attenuation and blurred/unstable boundaries; the intended cost path was fewer layer-compositing operations and less redundant geometric representation.

**NO-GO for originality.** [3D Surface Splatting for Inverse Rendering / 3DSS, 2605.05876v3](https://arxiv.org/html/2605.05876v3), sections 3.2–3.5, already does depth-interval grouping of surfels, accumulated EWA-weight normalization, coverage-derived per-layer opacity, and multi-layer compositing. Its formulation explicitly says visibility cannot be assigned independently to each surface sample. It differs from 2DGS by carrying no per-surfel learnable opacity and derives coverage from collective reconstruction weights. This is a direct mechanism collision, not merely similar motivation. Do not relabel it cohort opacity or correlated coverage, and do not ask Luna to implement it as an original candidate.

The candidate did not yet have measured cost benefit: grouping still evaluates each contributor, may add interval handling and memory, and cannot skip physically interleaving layers without changing rendering. Three-metric improvement and complete training time were hypotheses, not results. The source audit supplied a sufficient rejection before those experiments.

## Other strong boundaries found

| Source | Mechanism already present | Limit of this audit |
|---|---|---|
| [3D Skew-Normal Splatting, 2605.15010v2](https://arxiv.org/html/2605.15010v2), sections 3.2–3.4 | Gaussian times a Gaussian CDF, bounded learned asymmetry approaching a half-Gaussian, affine/marginal closure, sharp-boundary compactness, signed opacity. | Inspected formulation; no implementation or speed reproduction. One-sided support and negative density are not fresh. |
| [XClipGS, 2608.07760v1](https://arxiv.org/html/2608.07760v1), section 3.2 | Exact half-space-restricted ray integral under affine EWA: ordinary projected Gaussian times a conditional Gaussian CDF with affine pixel argument; differentiable with respect to primitive and clipping plane. | Medical volume clipping context differs, but the proposed per-splat analytic clipping primitive is already covered. |
| [Prune Wisely, Reconstruct Sharply, 2602.24136v1](https://arxiv.org/html/2602.24136v1), section 3.3 | A primary Gaussian minus a shared-center pseudo-Gaussian, adding opacity/scale factors to improve boundary expressiveness after pruning. | No local reproduction. Negative-lobe boundary repair plus fewer primitives is occupied. |
| [StableGS, 2503.18458v3](https://arxiv.org/html/2503.18458v3), Floater Analysis and Dual Opacity GS | Identifies opacity-gradient pseudo-equilibria for floaters and adds a separate geometry-opacity rendering path to isolate geometry constraints from appearance. | Uses additional depth losses/priors; no unchanged-objective or cost proof for our setting. Nevertheless floater deadlock and dual-opacity separation are prior. |
| [Geometry Gaussians, 2606.05124v1](https://arxiv.org/html/2606.05124v1), sections 3.2–4 | Separate geometric and appearance opacity, geometry-specific supervision, transparency-aware treatment. | No same-protocol reproduction; generic geometry/radiance-opacity decoupling is occupied. |
| [Geometry Field Splatting, 2411.17067v2](https://arxiv.org/html/2411.17067v2), sections 3–4 | Revisits volumetric versus stochastic opaque-geometry splatting and geometry-field representation. | Background/method screening only; do not claim a detailed equivalence beyond the reviewed text. |
| [VoroTracing, 2608.17682v1](https://arxiv.org/html/2608.17682v1), sections 4–5 | Co-designs a nonoverlapping Voronoi representation, coherent ray traversal, compact appearance, surface-concentrated opacity and fixed-budget optimization. | Reported rendering speed is not our complete training wall. Replacing overlap with surface-concentrated space partitioning has strong prior. |
| [SolidGS, 2412.15400v1](https://arxiv.org/html/2412.15400v1), section 3.1 and supplement | Uses a shared trainable kernel exponent to consolidate surfel opacity and improve sparse-view geometry consistency. | Opaque interiors, stronger kernel support, and removing semitransparent tails are not unexplored ideas. |
| [Gaussian Billboards, 2412.12734v1](https://arxiv.org/html/2412.12734v1), section 3 | Spatial texture inside oriented Gaussian splats; discusses opacity-shape extensions and half-primitive representations. | Reinforces the existing within-footprint appearance rejection. |

[When 3DGS Recovers Real Surfaces, 2608.30054v1](https://arxiv.org/html/2608.30054v1) analyzes geometry/appearance ambiguity under a first-hit abstraction. Its authors report opaque-billboard failures in synthetic high-angular-capacity tests but not their standard real-scene capture tests. It does not establish that our room/garden quality deficits are billboard failures; do not use the abstract theory as a scene diagnosis.

## Search and handoff

Versioned HTML/TXT files and arXiv query XML are under `astra_visibility_replacement_prior_20260928/`; `source_manifest.json` records hashes and URLs. Queries covered opaque/visibility partitioning, clipping, decoupling, signed/negative kernels, skew/asymmetric support and correlated surface coverage. This was a bounded method-text search, not an exhaustive originality certification. No positive novel candidate was established.

Keep the active correlated-direction Stage0 unchanged. A failure there does not justify reopening these rejected lines by combining or renaming their components. Root will consolidate confirmed prior-art facts and the no-go decision into D:\学习 and its changelog; this agent avoids concurrent writes to shared knowledge files.
