# P13-X Discovery — Preregistered Feasibility Study

## 0. Status
- Phase: P13-X
- Goal: determine whether a new 3DGS training execution model has enough headroom for >3x E2E speedup over the frozen P12 C0 + fused-SSIM strong baseline.
- This phase is discovery/oracle only. No headline speedup claim is allowed.
- No threshold tuning after observing probe results.

## 1. Frozen baseline identity
- gsplat upstream: nerfstudio-project/gsplat @ 512d366b67073d77ca099ede742683c165dfc23b
- C0: frozen C0_V3_FINAL30K = F9 + SCALAR_ADJOINT + H8_MR, PX=2
- C0 extension sha256: d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11
- C0 core sha256: 935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e
- fused-SSIM source sha256: 7b3784de48096f3b5d1ee156ce721f71f77a02650131f6c539331bc7d2d1ecc6
- fused-SSIM binary sha256: dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb
- recipe: seed=42, 30K, eps2d=0.3, tile=16, 1080p, packed=false, classic rasterization, black background
- loss: 0.8*L1 + 0.2*(1-SSIM), 11x11 Gaussian sigma=1.5, same padding.

## 2. Research hypotheses
H-D (Delta execution): after topology freeze, revisiting a fixed training camera exhibits sufficiently stable Gaussian/tile dependency and depth ordering that a persistent view program can update only a minority of work.

H-A (Compressed adjoint): the image-space loss adjoint can be represented by a low-dimensional tile basis while preserving the induced Gaussian parameter gradient sufficiently well to reduce rasterizer backward work.

These are separate hypotheses. Failure of one does not imply failure of the other.

## 3. Existing P12 evidence (X0, no new experiment)
From P12 C0 trajectories, densification stops at 15K and Gaussian count is constant from 15K to 30K in all 13 scenes. Across the same interval, the unweighted mean visible-Gaussian fraction is about 0.293 per sampled training view. This supports testing persistent topology and active worksets, but does NOT establish membership/order stability.

## 4. New probe scenes
Primary discovery triad: room, bicycle, garden.
Rationale: continuity with prior P12 preflight and different Gaussian populations.
If either H-D or H-A passes on all three, run one adversarial confirmation scene with high late visible fraction (train) before implementation.

## 5. Probe X1 — fixed-camera visible-set stability
For each scene select 8 deterministic anchor training cameras from the true COLMAP training set by evenly spaced sorted-camera indices. Do not use synthetic cameras.
From iteration 15K through 30K, every 250 iterations run diagnostic projection/rasterization on anchors under torch.no_grad(). Diagnostics are excluded from timing.
Record exact visible Gaussian IDs from the actual frozen renderer path.
Metrics for each anchor and lag {250, 500, 1000, 2500} iterations:
- Jaccard(visible IDs)
- retained fraction |A∩B|/|A|
- entered fraction |B-A|/|B|

H-D X1 PASS prerequisite: median Jaccard at lag=250 >= 0.95 on each primary scene.
X1 hard DROP for exact delta-rendering: any primary scene median Jaccard at lag=250 < 0.85.

## 6. Probe X2 — exact tile-membership and depth-order stability
Must use renderer-produced intersection/tile data or a debug exposure of the exact CUDA intermediates. Python geometric approximations are forbidden as evidence.
For each anchor snapshot record compact hashes/counts of exact (tile_id, gaussian_id) memberships and per-tile sorted Gaussian IDs. Raw dumps may be sampled; all summary metrics must be reproducible.
Metrics:
- membership Jaccard for (tile_id, gaussian_id)
- percentage of tiles whose membership hash is unchanged
- normalized inversion rate among common Gaussian IDs per tile
- percentage of tiles whose full order hash is unchanged

H-D X2 PASS prerequisite at lag=250:
- median membership Jaccard >= 0.95
- median order inversion rate <= 0.02
X2 hard DROP for exact cached sort reuse if membership Jaccard < 0.85 or inversion rate > 0.10 on any primary scene.

## 7. Probe X3 — dirty-work lower bound
Do NOT define a Gaussian as dirty merely because Adam touched its parameter buffer. Define observable dirty work relative to a fixed anchor view.
For each adjacent diagnostic snapshot, compare exact projected Gaussian attributes used by rasterization (mean2D/conic/opacity/color/depth/radius) for common visible IDs.
Evaluate tolerances eps in {0 exact-bitwise, 1e-6, 1e-5, 1e-4 relative L2 per attribute group} without choosing after the fact.
A tile is dirty when any member's renderer-input change exceeds eps, membership changes, or order changes.
Report dirty-tile fraction and dirty-intersection fraction for every eps.

H-D promising gate: at eps=1e-5, median dirty-intersection fraction <= 0.30 on all primary scenes AND exact-image replay error from reusing clean work <= 1e-6 relative L2.
H-D hard DROP as a >3x headline path: dirty-intersection fraction >= 0.70 at eps=1e-5 on any two primary scenes.

## 8. Probe X4 — tile adjoint compressibility
Use the exact P12 fused loss. Capture dL/dRGB for real training iterations after 15K. Sample 256 iterations/scene, deterministically spaced; tile size 16x16.
Use a fixed orthonormal 2D DCT basis per channel. Evaluate K in {4, 8, 16, 32, 64, 128, 256} low-frequency zig-zag coefficients. No data-dependent top-K for the primary result.
Report:
- adjoint energy retention
- adjoint relative L2
- cosine similarity
- percentage of tiles meeting each error threshold
This probe alone cannot validate H-A because rasterizer Jacobians are smoothing operators.

## 9. Probe X5 — induced Gaussian-gradient fidelity
On fixed renderer states (no optimizer step), compute exact Gaussian gradients from exact dL/dRGB and gradients induced by DCT-K reconstructed dL/dRGB.
Compare parameter groups separately: means, scales, quats, opacity, SH/color.
Metrics per K:
- relative L2 gradient error
- cosine similarity
- sign agreement for top-10% |gradient| entries
- downstream one-step loss delta using a shadow optimizer state (state cloned; no contamination of training)

H-A PASS candidate at K<=32:
- cosine >= 0.995 for means and opacity
- relative L2 <= 0.05 for means and opacity
- top-10% sign agreement >= 0.98
- one-step loss delta relative to exact <= 0.5%
All conditions must hold on all primary scenes.
H-A hard DROP: no K<=64 reaches cosine 0.99 for means on at least two primary scenes.

## 10. Probe X6 — oracle timing
Only after X1-X5 are collected. No real implementation claim.
Measure current C0 component times with CUDA events on the same frozen states.
Construct two independent upper bounds:
- Delta oracle: skip all work classified reusable by X2/X3, charge measured unchanged overheads.
- Adjoint oracle: replace pixel-adjoint interactions by measured K-basis interaction count and charge actual basis transform cost.
Also report composed oracle, but do not assume multiplicative independence; account for overlap.

GO to implementation only if:
- at least one hypothesis has a standalone predicted E2E speedup >= 2.0x, AND
- composed conservative oracle >= 4.0x over C0, AND
- no quality/correctness gate failed.
Otherwise P13-X verdict = DROP or INVESTIGATE, not implementation.

## 11. Required artifacts
P13X_SOURCE_IDENTITY.json
P13X_X0_P12_PRIOR.json
P13X_X1_VISIBLE_STABILITY.json
P13X_X2_MEMBERSHIP_ORDER.json
P13X_X3_DIRTY_WORK.json
P13X_X4_ADJOINT_DCT.json
P13X_X5_PARAM_GRAD_FIDELITY.json
P13X_X6_ORACLE.json
P13X_CONTAMINATION.json
P13X_VERDICT.json
P13X_REPORT.md
MANIFEST_P13X.json

## 12. Scientific restrictions
- No synthetic cameras.
- No approximate CPU footprint reconstruction as evidence for X2/X3.
- No changing thresholds after seeing results.
- No 30K multi-scene production campaign before oracle gate.
- No novelty claim before final prior-art audit.
- Exact and approximate variants must be labeled separately.
- Timed runs must exclude diagnostics and must be paired/back-to-back under the same contamination rules as P12.
