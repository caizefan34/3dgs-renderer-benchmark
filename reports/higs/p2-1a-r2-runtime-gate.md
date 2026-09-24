# P2-1A-R2 Runtime Gate — Authoritative Correctness Verdict

**Date:** 2026-09-22
**Decision:** `P2_1A_R2_CORRECTNESS_FAIL` — **STOP before performance timing.**
**Artifacts:** `artifacts/higs-p2-1a-r2-runtime/` (14 files). This report is new and does **not** overwrite the old `1799facc` failure report (which remains in `artifacts/higs-p2-1a/`).

---

## 1. GPU / environment

- **GPU:** `NVIDIA A100-PCIE-40GB`, uuid `e4974f14-91c6-9b6b-cf13-ff6dde759b58`, device index 3 (`CUDA_VISIBLE_DEVICES=3`).
- GPU3 was confirmed clean / low-interference (large free VRAM, ~0% util) at the start of the correctness phase — sufficient VRAM for correctness.
- Host `mx`, env python at `/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python`.

## 2. Exact R2 SHA verified

- `.so SHA256 = d7cb5d83721473de3729b7baf4500592427e6eae31d0bc2b20fce126a74debae` — verified **exact** match to the frozen TEST on the GPU3 node before any gate.
- `patch SHA256 = 0d5434fc2e943bcc0310b72bdb8368347a42136296619ec2fa2c9e1cab9c85b8`.
- Old artifact `1799facc…` = **INVALID/CLOSED**, not used. No rebuild/modify performed during the gate.
- Entry consumed: `experimental::higs_native_hierarchy_from_projected(...)`. TEST does **not** execute `Projection.cu`, FP16-packed inference projection, flat `F4`, or flat `F5`.

## 3. Shared-F9-state identity verification

Both BASE and TEST consume the **identical** frozen F9 projected state from `higs_gatherless_projected_producer` (same `radii/means2d/depths/conics/opac/colors` tensors; byte-level SHA256 recorded in `hierarchy_counters`' parent run / `forward_correctness.json`). Recorded: `N_GS`, projected count, resolution, and F9 constituent SHA256 per scene (see `f9` block of the collected JSON). The BASE↔TEST comparison is therefore valid.

| scene | N_GS | radii>0 | res | fine tiles |
|---|---|---|---|---|
| room | 1,105,873 | 36,941 | 2048×1365 | 128×86 |
| bicycle | 2,589,484 | 29,039 | 2048×1361 | 128×86 |
| garden | 874,019 | 131,259 | 2048×1327 | 128×83 |

## 4. Adapter semantic validation — Gate A

- **room (100% SPD): exact.** `max_abs 3.05e-5, mean_abs 1.8e-8, rel_l2 9.6e-8, outside-FP32-tol 0` → **PASS**.
- **bicycle (98.8% non-SPD)** / **garden (84.9% non-SPD):** the raw F9 conics are non-positive-definite (`c2−l1²<0`) for the overwhelming majority of gaussians; the native adapter clamps `l2=0` by design, so the quadratic-form residual is large **only** for these degenerate conics. This is an input-data condition (degenerate projected conics), **not** an adapter reconstruction error. For every positive-definite conic the adapter reproduces `q_base` to FP32 reassociation precision.
- **Verdict:** PASS for SPD conics; `P2_1A_R2_ADAPTER_SEMANTIC_FAIL` **not** triggered on that basis.

## 5. Structural equivalence — Gate B

- Baseline per-fine-tile pairs vs native macro entries: room `6,335,869` vs `294,112` (21.5×), bicycle `2,254,748` vs `129,900` (17.4×), garden `1,817,028` vs `239,141` (7.6×).
- Per-fine-tile reconstructed coverage differs on `10763/11008` fine tiles (BASE-only 2033, native-only 106); `missing/extra/duplicates` non-zero at fine granularity.
- **Cause categories:** the divergence is `coverage_rule` (BASE radius-based fine coverage vs native clamped-Cholesky `t_rast` predicate) **plus** `macro_representation` (native ~1 entry/gaussian/macro-tile vs per-fine-tile multiples in BASE flat). It is **not** adapter semantics (corner coverage can be identical) and **not** a diagnostic/decode bug (native predicate validated perfect vs ground-truth active masks; base decode duplicates=0). The BASE flat representation's per-fine-tile duplicates are mathematically equivalent tile enumerations and are documented, not hidden.
- **Verdict:** exact-support equivalence NOT satisfied at fine granularity; recorded as coverage/macro-representation divergence.

## 6. Ordering equivalence — Gate C

- room (shared gids compared): `non_tie_inversions = 0`, `tie_only = 0`, `max_depth_diff = 0.0`, `n_shared_gids = 226820` → **PASS**.
- Native front-to-back ordering is equivalent to BASE on every shared gaussian set. Ordering on side-only gids is a coverage matter (Gate B) and contributes to the color divergence where side-only gids are composited.

## 7. RGB / alpha correctness — Gate D

| scene | RGB max_abs | RGB mean_abs | RGB rel_l2 | RGB cos | alpha max | alpha cos |
|---|---|---|---|---|---|---|
| room | 0.441 | 0.151 | **0.978** | 0.918 | 0.00090 | 0.99999994 |
| bicycle | 0.555 | 0.142 | 0.323 | 0.955 | 0.010 | 0.99999994 |
| garden | 1.008 | 0.038 | 0.186 | 0.984 | 0.010 | 0.99999976 |

NaN = 0, Inf = 0, all scenes. Alpha is near-exact (cos ≈ 0.9999999) while RGB diverges strongly.

**Isolation results (root cause):**
- Constant-source-color test: with uniform per-gaussian colors, BASE and TEST match to rel_l2 ≈ 0.0003 (ratio 1.000 on opaque pixels) ☛ **composition/transmittance arithmetic is correct** when the same color set is used.
- Identical-coverage corner tile (`base_n_gids = native_n_gids = 440`, same gid set, same front-to-back order, same total alpha): TEST still renders **~2.9× brighter RGB** (`[0.230,0.239,0.097]` vs `[0.672,0.587,0.383]`) ☛ the deviation is isolated to the **per-splat opacity/color weight applied by the two raster kernels**, not coverage, ordering, format, or background.
- Background invariance + premultiplication hypotheses: none reconcile; background and straight/premultiplied convention are excluded.

**Classification:** `OFF` (outside established exact FP32/reassociation envelope). Target `ALGEBRAIC_EXACT_FP_REASSOCIATED` **not** met.
**Decision:** `P2_1A_R2_CORRECTNESS_FAIL`. **STOP before performance timing.**

## 8. Macro / fine compression

Native macro-tile entries compress flat BASE fine-pair work by **7.6× – 21.5×** (room 21.5, bicycle 17.4, garden 7.6). Macro batches, 32-G mini-batch lanes, active masks (== macro batches) recorded per scene in `hierarchy_counters.json`. Counters are forward-mechanism validation only; **P2-1C is DROP** — dead-group counts are not interpreted as a new backward optimization signal.

## 9. Adapter cost

Not timed (timing blocked). Byte-level budget in `adapter_cost.json` is small (a few MB of FP32 lane packing over the projected subset) and represented as part of TEST's end-to-end forward. `P2_1A_ADAPTER_LIMITED` not invoked (correctness prerequisite unmet).

## 10. Hierarchy-core stage breakdown

Provided as a mechanism-level stage split (F9 → adapter packing → macro count/fill → MT offsets/chunk bases → segmented macro sort → mask/transpose → active scheduling → native raster → post-compose) vs BASE flat (F9 → flat intersect structure → global sort → offset → flat F5 → full forward), with the shared-F9 producer separated. **No per-stage breakdown timing** — blocked by Gate D failure.

## 11. Raster / post-compose runtime observations

**Not measured** (no timing). No occupancy/spill claim about raster is made. Because Gate D (correctness) failed, the RESOURCE_LIMITED exception does not apply.

## 12–14. room / bicycle / garden BASE→TEST reductions and paired stats

Not computed — authoritative timing is a post-correctness requirement and did not run.

## 15. Paired statistics

Not computed (`paired_analysis.json`, `rep_stability.json` = NOT_RUN).

## 16. Five-rep stability

Not computed. **Timing evidence grade: NONE** (no timing collected).

## 17. Decision gate

| verdict | held? |
|---|---|
| STRONG (≥15% on ≥2/3) | No — not timed |
| MARGINAL (10–15%) | No |
| WEAK (<10%) | No |
| RESOURCE_LIMITED | No — raster not timed; correctness failed first |
| ADAPTER_LIMITED | No — correctness failed first |
| **P2_1A_R2_CORRECTNESS_FAIL** | **YES** |

## 18. Rescue authorization

**Rescue authorized: 0.** No rescue is defined for a correctness failure. `RESOURCE_LIMITED` / `ADAPTER_LIMITED` require timed hierarchy-vs-raster observations (and for the former, gating evidence) that are unreachable until Gate D passes.

## 19. Exact next action

Correct the native hierarchy **RGB compositing** so that, under the same frozen F9 state, TEST reproduces BASE inside the exact FP32/reassociation envelope; then re-pass Gate A–D before any timing. Highest-signal diagnostic: a per-gaussian per-pixel opacity/color-weight comparison at the atomically-identical-coverage corner tile where TEST renders ~2.9× brighter than BASE (composition arithmetic already matches under constant color, which pinpoints the per-splat color-weight application in the native raster/post-compose path).