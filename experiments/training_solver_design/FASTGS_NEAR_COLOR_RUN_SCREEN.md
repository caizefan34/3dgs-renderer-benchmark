# FastGS near-color accepted-hit run screen

**Decision: VETO; do not build new CUDA or start fixed-group/real-training follow-up from this opportunity.** On four room@4100 training views, even the loosest predeclared threshold (`4/255`) put `93.16%` of accepted hits in greedy same-color runs of length at most 2, while only `1.64%` belonged to runs of length at least 8. Every view independently crosses the veto threshold.

## Coverage and method

The read-only probe re-rendered the same fixed 256 uniformly sampled full-resolution pixels per view as `fastgs_order_conflict_probe.py`, using its seed `20260927` and camera indices `0, 17, 54, 121`. These are training cameras: DSCF4668, DSCF4687, DSCF4729, and DSCF4806. The in-memory COLMAP loader filtered to these four known training filenames before opening images; no held-out/test image files or camera objects were loaded. The model is room@4100, source commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`, PLY SHA-256 `64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc`.

For every sampled ray, the script replayed the native tile-list order and the forward gates (`power <= 0`, `alpha >= 1/255`, and the `next_T < 1e-4` stop). It excludes the stop-triggering candidate but includes all accepted hits before that stop. This covers all `1,024/1,024` sampled rays, including `599` early-stopped rays. Replayed `n_contrib` exactly matched the native buffer for all rays; every recomposed RGB matched the native render within `1e-4` (largest observed channel error `3.51e-7`). The four view early-stop counts were `136`, `154`, `177`, and `132`.

Runs are greedily extended in each ray's accepted native order while every channel's RGB range stays within the threshold. RGB is the camera-evaluated per-Gaussian feature captured from the native projection buffer, before alpha/transmittance weighting. `P_len<=2` and `P_len>=8` are hit-weighted: they count hits belonging to runs of the indicated length, divided by all accepted hits. The native order is the rasterizer's tile depth-list order, not a newly computed per-ray Mahalanobis depth sort.

## Results

| RGB channel-range threshold | Pooled `P_len<=2` | Pooled `P_len>=8` | Accepted hits |
|---:|---:|---:|---:|
| `1/255` | 95.72% | 1.41% | 52,266 |
| `2/255` | 95.31% | 1.50% | 52,266 |
| `4/255` | 93.16% | 1.64% | 52,266 |

At `4/255`, the per-view hit-weighted (`P_len<=2`, `P_len>=8`) results were:

| Training view | Accepted hits | Early-stopped rays | `P_len<=2` | `P_len>=8` |
|---|---:|---:|---:|---:|
| DSCF4668 | 12,434 | 136/256 | 91.99% | 2.16% |
| DSCF4687 | 13,919 | 154/256 | 92.20% | 2.46% |
| DSCF4729 | 13,487 | 177/256 | 92.38% | 1.66% |
| DSCF4806 | 12,426 | 132/256 | 96.24% | 0.20% |

The predeclared `4/255` veto condition (`P_len<=2 >=70%` or `P_len>=8 <30%`) holds pooled and on each view, so this screen stops here.

This is an optimistic upper bound for per-ray adaptive grouping: it lets each ray choose groups using that view's current RGB values. It does not show that shared anchors or fixed groups preserve quality, save time, help training, or are original. No training or author-code changes were made. The previous `fastgs_order_conflict_result.json` does not contain per-hit RGB sequences; the new script captures them and independently validates native coverage before calculating runs.

The raw ray-level results, including accepted Gaussian IDs/colors/alphas, run lengths, `n_contrib` comparisons, parity, and early-stop status, are in `fastgs_near_color_run_result.json`. Probe SHA-256: `65d103b8fd3041925adde786af5e2cc2a82b863d425fa556cc8c7b87bf37233d`; result SHA-256: `bcbbfbbe3e6a5de263a8d5f73f242c9c90a009d8fed60e7dd6c63ed412540375`. It ran with `CUDA_VISIBLE_DEVICES=6` on EPIC-05; GPU6 was idle before launch and returned to `0%` utilization / `0 MiB` after exit.
