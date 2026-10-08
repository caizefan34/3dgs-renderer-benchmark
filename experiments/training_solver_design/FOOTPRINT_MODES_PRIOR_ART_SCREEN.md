# Within-footprint appearance modes: bounded prior-art screen

2026-09-27. **Reject the proposed Gaussian-envelope plus local Hermite/affine color-mode primitive as a novel 3DGS core.** This was considered as a way for one splat to fit more high-frequency detail, perhaps reducing overlap, backward work and training wall while improving quality. No implementation or GPU experiment was run.

[Resonant Brane Splatting for Arbitrary-Scale Super-Resolution](https://arxiv.org/html/2606.29453v3) (RBS), §3.2, explicitly uses a Gaussian envelope with internal Gaussian–Hermite modes and a separate RGB coefficient for each mode. Its abstract and §4.2 state the same causal proposal: internal color variation reduces the number of overlapping primitives required for fine detail and speeds rasterization. RBS is a **2D feed-forward super-resolution** method, so its reported results cannot be transplanted into 3D multiview training or compared against FastGS-Big/ACE-GS. It nonetheless occupies the proposed representation and speed rationale; simply making it view-consistent in a 3DGS trainer would need a new independently demonstrated technical idea before an originality claim.

Earlier 3DGS/2DGS prior art also narrows the space: `TOPOLOGY_HOMOTOPY_DESIGN.md` and `NEXT.md` already record Textured-GS, SVGS, Textured Gaussians and LiteTex-GS for spatially varying splat color; `MOMENT_FOURTH_SPLIT_SCREEN.md` records 2DGH for Gaussian–Hermite opacity/support modes. This screen does not prove all possible expressive primitive designs are known.

The cost hypothesis has an unmeasured break-even condition. If each surviving contribution costs `1+δ` times a standard contribution and the new method retains a fraction `r` of accepted hits, the hit-proportional component only falls when `r(1+δ)<1`. This is necessary even before added attribute state, event overhead, changed convergence, and complete process time. Native FastGS's per-Gaussian constant RGB interface cannot represent per-pixel internal colors with one unchanged forward/VJP call; an exact 3D implementation would require a new raster path and parity checks. Higher-order opacity modes additionally change support, alpha thresholds, early stopping and backward semantics. No real `δ`, `r`, image quality, training time, or unseen-view metric was measured here.

## Source identity

- Local RBS HTML snapshot: `topology_sources/rbs_2606_29453v3.html`, SHA-256 `454dc0ad0823e31393d5a3d527d2a0d0e252291bce3c5cfef86c07dfea1df4c6`.
- Direct source anchors: abstract; §3.2, equations (2)–(4); §4.2 “Internal expressiveness minimizes overlap and accelerates rendering.”
- This is a bounded prior-art rejection of the specific representation/rationale, not a 3D training performance result or a universal novelty search.
