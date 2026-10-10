# PRIOR_ART_MATRIX — GSSTREAM-P0 (Gate G0)

Preregistered literature gate for the GSSTREAM-P0 falsification kit. CPU-only review.
Date 2026-10-10. No GPU. Sources verified by direct fetch; where a claim could not be
independently verified it is marked UNVERIFIED.

Hypothesis under test: a **GPU-cheap compressed support index** for a differentiable
rasterizer. Per projected Gaussian it derives the exact per-ROW pixel support interval
(scanline ellipse interval) and proposes to *generate row spans during binning, retain
rank-stable warp-block segments, and consume them as a compressed ordered stream in BOTH
forward and backward, without per-pixel O(A) global list materialization.*

## Verdict: **G0 = NOVELTY_FAIL** (STOP_NOVELTY)

The central "build a compressed ordered support structure during binning and consume it
in both forward and backward" pattern is **already published** on the raster path in
gsplat's `packed=True` / `sparse_grad=True` active-tile forward/backward, and
structurally in 3DGART. The kit's own concede understates the collision.

| # | Article | Source / URL | Mechanism | Fwd/Bwd | Exact/Approx | Overlap with claimed mechanism | Claimed difference | Novelty verdict |
|---|---------|--------------|-----------|---------|--------------|-------------------------------|--------------------|-----------------|
| 1 | Speedy-Splat / AccuTile (Hanson et al., CVPR 2025, pp. 21537–21546; arXiv:2412.00578v3) | speedysplat.github.io; arxiv.org/abs/2412.00578 | **SnugBox** tight AABB + **AccuTile** per-row/column exact ellipse→tile intersection (2 pts/scanline); α=1/255 cutoff ellipse extent | Forward binning/culling only | Exact per-row | **Direct hit** on "exact per-ROW support interval from the conic quadratic form" + row-span culling | AccuTile is forward-only; no reusable per-row stream in backward | Mechanism published; only backward-reuse unclaimed (claimed by #3/#5) |
| 2 | FlashGS (Feng et al., **arXiv:2408.07967, Aug 2024**) | arxiv.org/abs/2408.07967 | Precise opacity-aware ellipse→tile intersection (3 redundancy eliminations) + pipelining | Forward | Exact-ish (AABB/tile, not per-pixel) | Confirms opacity-aware precise culling is known; same α-threshold ellipse idea | None | **NOVEL** — but see CITATION-INTEGRITY FLAG below |
| 3 | gsplat `isect_tiles` + `rasterize_to_indices_in_range`, packed/sparse active-tile fwd/bwd | docs.gsplat.studio/main/apis/*; cuda/_torch_impl.* | `isect_tiles` → per-(image,tile) sorted intersection IDs (image_id\|tile_id\|depth) + `flatten_ids`; `packed=True`+`sparse_grad` → COO sparse grads; **Aug-2026 main absorbed AccuTile/SnugBox opacity-aware culling into `isect_tiles`** | **Both** (RegisterRasterizeToPixelsSparse bwd) | Exact per-tile ellipse culling; indices are per-TILE, NOT per-row, NOT per-pixel | **Strong hit on build-and-consume in BOTH passes + no O(A) dense global list** (packed/sparse already ships this) | Index is per-TILE not per-ROW; per-ROW rank-stable segment stream is the only unclaimed sliver | **Mostly published.** Pattern occupied; residual = per-ROW vs per-TILE granularity only |
| 4 | 3DGART (Vit, Batchelor, Green — arXiv:2608.17298, Aug 2026) | arxiv.org/html/2608.17298 | Ray-traced Gaussian; compact primitive-major tile-local pixel-local ordered buffer (L,T); prefix-sum primitive offsets; backward = one thread/primitive/tile, gather over covered pixels, single atomicAdd | Backward (forward is RT) | Exact | **Closest structural collision** on the consumed-structure side: ordered, rank-stable, compressed intermediate stream built at binning and consumed by gather-style backward without per-pixel global scatter | **Ray-traced, NOT the raster path**; builds per-pixel (L,T) buffer not per-row spans | Raster-vs-ray-traced forward differs; pattern (compact ordered buffer + primitive-centric gather bwd) published |
| 5 | StopThePop (Radl et al., SIGGRAPH 2024 / ACM ToG 43(4); arXiv:2402.00525) | r4dl.github.io/StopThePop/ | Tile-based culling + per-tile depth re-sorting; view-consistent ordering | Forward culling + per-pixel sort | Exact per-pixel depth, per-tile culling | Confirms per-tile culling + reordering published | None on compressed backward stream | **NOVEL** on the compressed-stream-reused-in-bwd claim (not a collision) |

## G0 decision reasoning

Preregistered G0 rule: *"an existing paper/code already performs the claimed same
compressed build-and-consume mechanism ⇒ STOP_NOVELTY regardless of work counts."*

- **Same mechanism, already published:** gsplat `packed`/`sparse_grad` active-tile
  forward/backward (#3) ships "sparse index reused in both passes with no dense
  `[C,N,…]` global list" on the raster path. This is the kit's core claim, already
  shipping.
- **Same backward-buffer pattern:** 3DGART (#4) publishes the ordered-compressed-support
  buffer + gather-style backward. It differs only by ray-traced vs raster forward, so it
  is not an exact same-mechanism hit, but it removes the "no one has shown the backward
  buffer pattern" escape.
- **Per-ROW math is published:** AccuTile/Speedy-Splat (#1) already publishes exact
  per-scanline ellipse interval derivation. The kit's row-span derivation is not new math.

**Strongest remaining novelty argument (for the record):** the single defensible sliver is
*granularity + ordering* — a **per-ROW** interval stream with **rank-stable warp-block
segment** ordering, finer-grained than gsplat's per-TILE `isect_ids`, *also* used as the
compressed backward access pattern. This is an incremental refinement of a shipped feature
(per-tile → per-row), not a new falsifiable mechanism, and it would have to beat gsplat's
packed baseline on build bytes / sort cost / backward global traffic to survive. As
written, the mechanism is occupied ⇒ **NOVELTY_FAIL**.

**Strongest prior-art collision:** gsplat `isect_tiles` + `rasterize_to_indices_in_range`
packed/sparse active-tile forward/backward (#3), which on current main has also absorbed
AccuTile/SnugBox — a single source occupying exact ellipse row/tile culling + sparse-index-
reused-both-passes.

## Citation-integrity flag

The kit (README.md / DSH_TASK.md) cites **"FlashGS CVPR 2025: FlashSplat / precise
Gaussian intersection"**. The real FlashGS (arXiv:2408.07967) is **Aug 2024**, and has **no
"FlashSplat" component and no multi-resolution rendering mode**. The kit appears to cite a
different or mislabeled work. This must be corrected before any claim is registered.
(Also note: the kit's "gsplat main Aug 2026 already has active-tile forward/backward with
sparse IDs" is consistent with #3 and is the load-bearing collision.)

## Caveats on this review

- `web_search` was unavailable this session; findings come from direct `web_fetch` of
  primary sources (arXiv abstract/HTML, project pages, gsplat docs/source). A targeted
  sweep for "scanline/interval compressed Gaussian list reused in backward" (HACM,
  Turbo-GS, GOF, etc.) was not exhaustively run; nothing encountered suggests a per-row
  compressed-stream-reused-in-backward rasterizer exists.
- 3DGART (arXiv:2608.17298) is future-dated relative to typical citations; treated as the
  authors' preprint per its arXiv record.
