# MSB0-P0 Report — Minimal Sufficient Backward State

**Verdict: `DROP_WORK_AMPLIFICATION`**
Date: 2026-10-01 · Track: P0 THEORY / SOURCE / EXACT REPLAY ORACLE (no CUDA, no training)

## 1. Research question

Per-Gaussian backward in modern 3DGS (FasterGSFused) stores intermediate blending
state — here `bucket_color_transmittance = float4(C_b, T_b)` per (bucket,
tile-pixel) — to enable **independent per-bucket parallel reverse-mode** backward.
BSC0 asked whether those values can be *encoded with fewer bits*. MSB0 asks the
more fundamental question:

> Which saved state components are mathematically necessary, and which can be
> reconstructed exactly (or to baseline FP tolerance) from other already-available
> quantities? The target is state **elimination**, not quantization.

## 2. Summary answer

The saved `bucket_color_transmittance` state is **NOT redundant**. It is the
materialized **prefix** of the front-to-back compositing recurrence, and it is the
minimal-sufficient exact-replay summary:

- `C_b` (prefix accumulated color) and `T_b` (prefix transmittance) are
  **two independent, non-reconstructible scalars**. Dropping either breaks a
  distinct gradient family (Gate 4).
- The only exact reconstruction — rebuilding the prefix by a sequential
  forward-carry from the known constant `(C_0=0, T_0=1)` — is **bit-exact** but
  requires a **second full tile-Gaussian traversal** plus **global cross-bucket
  synchronization**, and is exactly the **original pixel-centric 3DGS backward**
  (Gate 5 / prior-art adjacency).

Candidates were fixed before replay (M0/M1/M2/M3); no post-hoc invention.

## 3. Gate outcomes (prereg applied literally)

| Gate | Result | Evidence |
|------|--------|----------|
| 0 Remote | PASS | All PyTorch/replay on EPIC-05 (A100-SXM4-80GB); local = control-plane only |
| 1 Source lock | PASS | faster-gaussian-splatting `ae665af2` (2026-09-19); ngs `e458ac6e`; kernels transcribed |
| 2 Prior art | PASS | No direct collision; M3 == original pixel-centric backward (adjacency, not novelty) |
| 3 Dependency | PASS | M1 73.3–73.6%, M3 97.7–98.2% reduction (>=25%); M2 24.4–24.5% (<25%) |
| 4 Fidelity | **M1/M2 FAIL** | M1 `dL_dalpha` cos 0.02/0.84/0.07; M2 `dL_dcolor` cos 0.45/0.68/0.25 |
| 5 Overhead | **M3 FAIL** | second traversal + global sync |
| 6 Economics | not reached | no candidate survives 4+5 |

**Terminal verdict = `DROP_WORK_AMPLIFICATION`.** M3 is the furthest-reaching
candidate (>=25% reduction AND bit-exact) and is eliminated at Gate 5.

## 4. Method (exact-replay oracle, float64, EPIC-05)

1. **Source lock.** Transcribed `blend_cu` (forward) and `blend_backward_cu`
   (backward) from the pinned FasterGSFused CUDA source. See `MSB0_SOURCE_LOCK.json`.
2. **Frozen camera.** For each scene, COLMAP image id 0; projected all Gaussians
   to the camera (`project_gaussians`, float64).
3. **Tile selection.** Chose the max-overlap tile (deterministic, deep
   compositing); capped at 2048 positions (nearest) for tractability.
4. **Forward.** `composite_tile` reproduces the front-to-back recurrence and saves
   per-bucket prefix state `(C_b, T_b)` exactly as `blend_cu` does every 32
   Gaussians.
5. **Backward.** `backward_tile` reproduces `blend_backward_cu` for each candidate
   mode M0/M1/M2/M3.
6. **Independent validation.** `autograd_ground_truth` builds a differentiable
   forward per tile and obtains gradients via PyTorch autograd; the hand-derived
   M0 backward is compared against it (`M0_vs_autograd`).
7. **Economics.** `msb0_economics.py` counts per-tile buckets over the full image
   and computes saved-state bytes and per-candidate reduction.

## 5. Replay fidelity (Gate 4)

Reference = M0 (faithful bucket-seeded backward). Gate-4 threshold: cos >= 0.999999,
rel-L2 <= 1e-5, no NaN/Inf.

**M0 vs autograd (validates the oracle itself):**

| Scene | dL_dcolor | dL_dmean2d | dL_dopacity | dL_dconic |
|-------|-----------|------------|-------------|-----------|
| room | 1.0000 | 1.0000 | 1.0000 | 0.9432 |
| garden | 1.0000 | 1.0000 | 0.9997 | 0.9729 |
| bicycle | 1.0000 | 1.0000 | 1.0000 | 0.9847 |

The compositing-critical gradients (`dL_dcolor`, `dL_dmean2d`, `dL_dopacity`) match
autograd exactly. `dL_dconic` differs only by the kernel's documented `0.5x`
cross-term convention (see `MSB0_SOURCE_LOCK.json` → `conic_convention_note`);
this is orthogonal to MSB0 because the conic gradient does not depend on the
saved compositing state.

**Candidates vs M0:**

| Scene | M1 dL_dalpha | M2 dL_dcolor | M2 dL_dalpha | M3 all |
|-------|--------------|--------------|--------------|--------|
| room | 0.0197 | 0.4498 | 0.0562 | 1.0000 |
| garden | 0.8393 | 0.6821 | 0.0614 | 1.0000 |
| bicycle | 0.0735 | 0.2452 | 0.0632 | 1.0000 |

- **M1** (drop `C_b`): `dL_dalpha` collapses → `C_b` not locally reconstructible.
- **M2** (drop `T_b`): `dL_dcolor` and `dL_dalpha` collapse → `T_b` not locally
  reconstructible.
- **M3** (forward-carry): exact everywhere (rel <= 2.2e-11) but Gate-5 violating.

## 6. Saved-state bytes and reduction (Gate 3 / Gate 6 reference)

Full image, frozen camera, 8160 tiles (1920x1080, 16px tiles).

| Scene | total buckets | bucket_ct bytes | M0 total bytes | M1 red% | M2 red% | M3 red% |
|-------|---------------|-----------------|----------------|---------|---------|---------|
| room | 172,087 | 704,868,352 | 721,457,152 | 73.28 | 24.43 | 97.70 |
| garden | 183,819 | 752,922,624 | 769,511,424 | 73.38 | 24.46 | 97.84 |
| bicycle | 219,685 | 899,829,760 | 916,418,560 | 73.64 | 24.55 | 98.19 |

`bucket_color_transmittance` (16 B/bucket-pixel) dominates; per-pixel buffers
(`tile_final_transmittances`, `tile_n_processed`, 4 B each) are ~256x smaller.
M1 removes the 12-byte RGB of each float4 (keep `T_b`); M2 removes the 4-byte
`T_b` (keep `C_b`); M3 eliminates `bucket_color_transmittance` entirely.

## 7. Overhead (Gate 5) — why M3 fails

The FasterGSFused design launches one block **per bucket**, and buckets are
**independent** — that parallelism is the entire point of saving the prefix
state. M3's forward-carry makes bucket `b+1`'s entering state a function of
bucket `b`'s exiting state, so:

- buckets must be processed **sequentially** (a prefix scan across buckets);
- a **second full tile-Gaussian traversal** is required to rebuild the prefix;
- **global cross-bucket synchronization** replaces the existing per-bucket
  boundaries.

This violates two explicit Gate-5 prohibitions. It also re-implements the
original pixel-centric 3DGS backward (prior-art adjacency) with no benefit.

## 8. Economics (Gate 6, reference only)

Gate 6 is not reached (no candidate survives Gates 4+5). For reference, M3's
peak-VRAM reduction if `bucket_color_transmittance` were eliminated =
7.86% / 6.77% / 8.28% (room/garden/bicycle) — all < 8% — and E2E would **regress**
due to the added traversal. Using the strong fused-backend timings
(room 260 s/iter, garden 523, bicycle 448 on A100), the backward's saved-state
memory traffic is a small fraction of iteration time and is compute/atomic
bound, not bandwidth bound — so even 100% elimination of the saved state cannot
reach the 1.04x E2E bar, and M3 makes it negative.

## 9. Conclusion and recommendation

- The saved `bucket_color_transmittance = float4(C_b, T_b)` is the
  **minimal-sufficient exact-replay state**. Both `C_b` and `T_b` carry
  independent information; neither is reconstructible from final outputs,
  current-bucket gaussians, or neighboring summaries without a cross-bucket
  prefix scan.
- **No state elimination is viable.** M1/M2 fail fidelity (Gate 4); M3 is exact
  but is work amplification + prior-art adjacency (Gate 5).
- **Recommendation:** do not pursue MSB0 elimination. The proven-non-redundant
  state is the correct input to **MSB1**, where BSC0-style compression
  (quantization / bit-reduction) should be applied to the **retained** state.
  MSB0 elimination does not compose with BSC0 because there is no redundant state
  to remove first.

## 10. Reproducibility

- Environment, source lock, and file inventory: `MSB0_TRACE_INDEX.md`.
- Verdict JSON: `MSB0_VERDICT.json`.
- Raw traces: `msb0_replay_{room,garden,bicycle}.json`,
  `msb0_economics_{room,garden,bicycle}.json`.
- Scripts: `msb0_oracle.py`, `msb0_run.py`, `msb0_economics.py`,
  `dump_results.py`, `run_econ.sh` (mirrored on EPIC-05 under
  `/root/msb0_p0/scripts`).
- Integrity: `MANIFEST.sha256`.
