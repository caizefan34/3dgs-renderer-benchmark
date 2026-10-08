# RF0B_REPORT — FastGS Density-Control Policy Port into Frozen C0

Task: DSH-3 ReproFast0 **Lane B**. Date: 2026-10-01. Host: EPIC-05 A100-80GB.
FastGS source: `fastgs/FastGS` @ `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f` (frozen independently by Lane B; reconciled with Lane A — same commit).

## 0. Question and verdict

**Primary question:** Does FastGS's *algorithmic* density-control policy retain a
large speed gain when renderer, loss, optimizer and training horizon are held
fixed?

**Verdict: NO.** Under a frozen C0 renderer, porting ONLY the FastGS
multi-view-consistent densification/pruning policy is a **net slowdown** on
2 of 3 scenes and a marginal gain on 1. Mean wall-clock speedup = **0.819x**
(i.e. c0fgs is 22% slower on average). The policy's genuine,
large contribution is **model compactness** — 85–94% fewer final
Gaussians and 79–91% smaller Gaussian-count integral — **not** speed. FastGS's
"fast" comes from its renderer/CUDA backend, which was explicitly NOT ported
(fail-closed rule). The multi-view scorer adds 10 extra render+loss passes per
densification event; that fixed overhead dominates any per-iter saving from a
smaller Gaussian set.

## 1. Is the policy port faithful?

**Yes, with three documented, forced approximations** (see `RF0B_POLICY_DIFF.md`
§4). All FastGS policy *decisions* are ported 1:1 — grad-qualified clone/split,
`importance_score > 5` metric mask, opacity+screen+world prune, 0.5-budget
multinomial prune, 0.8 opacity cap, `opacity<0.1 OR score>0.9` final prune every
3k in (15K,30K), schedule (`size_threshold=20 if iter>3000`), 10 score views,
`grad_thresh=0.0002`/`grad_abs_thresh=0.0012`/`dense=0.001`/`min_opacity=0.005`.
`final_N`, clone/split/prune counters, and holdout metrics all match the FastGS
spec. The three unavoidable mismatches (renderer cannot expose the contracted
signed screen gradient or the true per-Gaussian covered-pixel count) are
reconstructed with minimum bookkeeping and are **not tunable post-hoc**.

## 2. Per-scene C0-vs-policy speedup (wall-clock, 30K, seed42)

speedup = C0_wall / c0fgs_wall; >1.0 means the policy is faster.

| Scene | C0 wall (s) | c0fgs wall (s) | Speedup | Δ (s) |
|---|---|---|---|---|
| room | 513.6 | 712.7 | 0.721x | +199.0 |
| garden | 814.1 | 758.6 | 1.073x | -55.5 |
| bicycle (MipNeRF-360) | 866.9 | 1310.2 | 0.662x | +443.3 |

- room: **0.721x** (39% slower) — scoring overhead dominates the small scene.
- garden: **1.073x** (-7%) — only scene where N-reduction pays off mid/late.
- bicycle: **0.662x** (51% slower) — large scene, heavy early scoring cost.

### Stage-wall speedup (0–5K / 5–15K / 15–30K)

| Scene | 0–5K | 5–15K | 15–30K |
|---|---|---|---|
| room | 0.638x | 0.737x | 0.748x |
| garden | 0.789x | 1.148x | 1.140x |
| bicycle (MipNeRF-360) | 0.325x | 0.513x | 1.103x |

Reading: c0fgs is **slower in 0–5K on every scene** (scoring begins at iter 500)
and only catches up in 15–30K on garden/bicycle — after densification stops
(15K) the model is far smaller so each render is cheaper and only the every-3K
final-prune scoring remains. On room the model is already tiny, so the scoring
overhead is never repaid.

## 3. True-holdout quality deltas (i%8==0, never trained; seed42)

| Scene | ΔPSNR | ΔSSIM | ΔLPIPS |
|---|---|---|---|
| room | -2.602 | -0.0525 | +0.1189 |
| garden | -2.784 | -0.1773 | +0.2248 |
| bicycle (MipNeRF-360) | -2.429 | -0.1558 | +0.2339 |

All deltas are **negative for PSNR/SSIM and positive for LPIPS** (worse). The
policy costs **2.78–2.43 dB holdout PSNR** (a loss, not a gain). Absolute holdout:
room 30.80→28.20, garden 26.01→23.23, bicycle 21.70→19.27 PSNR. The aggressive
multi-view prune + final prune remove Gaussians the frozen C0 renderer would
have kept; combined with the box-vs-ellipse footprint over-count (POLICY_DIFF
§4.1/§4.3) the `importance_score>5` threshold bites harder than in FastGS's own
renderer. No threshold was retuned after seeing these results (fail-closed).

## 4. Gaussian-count and integral reduction

Gaussian-count integral = Σ_t N_t·Δt (time-weighted), with N_t the per-iter
Gaussian count and Δt from the eval-checkpoint wall-time axis.

| Scene | C0 N | c0fgs N | N red. | C0 ∫N dt | c0fgs ∫N dt | ∫ red. |
|---|---|---|---|---|---|---|
| room | 853,453 | 86,379 | -89.9% | 3.563e+08 | 7.337e+07 | -79.4% |
| garden | 2,548,992 | 371,809 | -85.4% | 1.847e+09 | 2.628e+08 | -85.8% |
| bicycle (MipNeRF-360) | 3,703,162 | 225,187 | -93.9% | 2.724e+09 | 2.408e+08 | -91.2% |

The policy yields a **85–94% smaller final model** and **79–91%
smaller Gaussian-count integral** — a large, real compactness win. Peak VRAM also
drops in every scene (c0fgs 5.16/5.50/5.24 GB vs C0 5.86/7.92/8.58 GB; −12%/−31%/−39%),
consistent with the smaller live Gaussian set. This is the honest core finding:
**the density-control policy is a compression mechanism, not a speed mechanism,
once the renderer is held fixed.**

## 5. Why the policy is slower under a frozen renderer

1. **Scoring renders are pure overhead here.** FastGS renders 10 random TRAIN
   views per densification event (every 100 iters in 500–15K) and per final-prune
   event (every 3K in 15K–30K) to compute `importance_score`/`pruning_score`.
   In FastGS these share its fused CUDA renderer; here they run on the frozen C0
   renderer with no amortization. This lands in the `densify` phase (≈1.17 ms/iter
   amortized) but inflates the `loss` phase to ≈7.3 ms (allocator churn from 10
   extra render+SepSSIM allocations/event; loss *values* are unchanged and match C0).
2. **The N-reduction only pays off late.** Per-iter cost ∝ visible Gaussians.
   c0fgs only pulls ahead after 15K (densify off), and only when the scene is
   large enough (garden/bicycle 15–30K speedup 1.14x/1.10x). For room the model
   is small throughout, so overhead is never repaid.
3. **FastGS's actual speedup is backend-side.** Its tile/culling CUDA kernels
   (not ported, per fail-closed) are what make it "Fast". Isolating the density
   policy removes that and exposes the scorer's cost.

## 6. Artifacts and verification

Local: `C:\Users\36570\3dgs-renderer-benchmark\rf0b_laneB\`
Remote: `/root/rf0b_laneB/deliverables/`

- `RF0B_ENVIRONMENT.json` — host, torch, gsplat, CUDA, env locks.
- `RF0B_SOURCE_LOCK.json` — FastGS @44e02a5 + C0 trainer/renderer sha256.
- `RF0B_CAMERA_SPLIT.json` — corrected i%8==0 holdout (311/39/272, 185/24/161, 194/25/169).
- `RF0B_POLICY_DIFF.md` — files changed, C0 semantics preserved, policy ported, mismatches, line mapping.
- `RF0B_RAW.json` — full per-scene raw (both arms): stages, quality, Gaussian integral + trajectory.
- `RF0B_QUALITY.json` — true-holdout PSNR/SSIM/LPIPS + deltas.
- `RF0B_TIMING.json` — full + stage walls + phase means (both arms).
- `RF0B_GAUSSIAN_TRAJECTORY.json` — per-iter N_t (sampled) + integral per arm.
- `RF0B_SUMMARY.json` — per-scene C0-vs-policy speedup, deltas, reductions.
- `RF0B_REPORT.md` — this file.
- `MANIFEST.sha256` — sha256 of every RF0B_* deliverable.

Integrity: all three c0fgs runs report `status=SUCCESS`, `all_losses_finite=true`,
`contaminated=false`, `timing_grade=PUBLICATION`, `loss=sep`, seed 42, 30000 iters.
No post-hoc threshold tuning, no algorithmic rescue, no local GPU, no backend
substitution. FastGS ideas are NOT claimed as our novelty.

## 6b. Intersection-count integral

Not available. The frozen C0 renderer does not expose per-Gaussian
intersection counts; counting them requires a custom intersection kernel
(Lane C / DPC0 territory), which is out of Lane B policy-port scope. See
`RF0B_RAW.json` -> `intersection_integral.available = false`.

## 7. Bottom line

Porting FastGS's algorithmic density-control policy into a frozen C0 trainer is
**faithful** but is a **net slowdown** (mean 0.82x) that buys **85–94% fewer
Gaussians** at the cost of **~2.5 dB holdout PSNR**. The policy is a **compression
tool**, not a speed tool, when the renderer/loss/optimizer are held fixed —
FastGS's speed advantage lives in its backend, which Lane B deliberately did not
port.
