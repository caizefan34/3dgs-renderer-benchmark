# A0-P10 Preflight Kit — v2.1 (lead-review revisions + remote-execution outcome)

**Campaign:** A0-P10_PREFLIGHT (native tile×batch census for the frozen P12/C0 baseline)
**Cell:** room@25K, frozen camera 19 (DSCF4819), top-32 native tile×batch buckets
**Verdict (this run):** `A0P_MEMORY_BLOCKER` — frozen-recipe OOM on EPIC-05 A100-80GB

---

## 1. The 8 lead-review revisions (all implemented in v2)

| # | Revision | Where | Status |
|---|----------|-------|--------|
| 1 | Correct `Common.h` constants | `a0preflight/lib.py` | **Fixed.** `ALPHA_THRESHOLD=1/255`, `MAX_ALPHA=0.99`, `TRANSMITTANCE_THRESHOLD=1e-4`, `MIN_ONE_MINUS_ALPHA=1e-6` (was `1.0e-20` / `1e-4`). Verified against the frozen CUDA source. |
| 2 | `T` init `T_final` | `a0preflight/lib.py` | **Fixed.** Blend-bwd `T` is initialized to `T_final = 1 - render_alphas[p]` (the frozen kernel's `T` accumulation invariant), then walks the batch. |
| 3 | Exact adjoint from `v_render_colors` | `a0preflight/lib.py` | **Fixed.** `q_color[p,i,k] = fac · v_render_colors[p,k]`, where `v_render_colors = autograd.grad(loss, raw pre-clamp frame, retain_graph=True)` on the same native graph (NOT a re-render). |
| 4 | Native bucket split | `a0preflight/lib.py` | **Fixed.** Buckets are native `(tile_id, batch_index b)` with width block_size = 256/PX = 128; `batch_end = range_end-1-b·128`; `batch_size = min(128, batch_end+1-range_start)`; `g(t) = flatten_ids[batch_end-t]`. |
| 5 | Correct constants | — | Same as #1. |
| 6 | `q_alpha` / `q_color` exact algebra | `a0preflight/lib.py` | Verified: `q_alpha = Σ_k (rgbs[k]·T − buffer[k]·ra)·vc[k]`; `q_color = fac·v_render_colors`; conic pairing `sigma = 0.5·(x·dx²+z·dy²)+y·dx·dy`. |
| 7 | `H8_MR` independence | `a0preflight/oracle_analyze.py` | **Documented.** H8_MR changes the *output* gradient path, not the per-pixel `Q` capture; the P10 census is on the exact `Q` (alpha + color joint), so H8_MR is a no-op for the verdict. |
| 8 | 3-channel `q_color` in the joint SVD | `a0preflight/oracle_analyze.py` | **Fixed.** The analyzer uses the REAL 3-channel `q_color` (not a scalar channel) for the joint Q-color SVD; `m ≤ 128` check; verdict via the frozen `p10_verdict`. |

---

## 2. Synthetic verification (ALL PASS)

- `tests/_test_p10.py`: `q_color` / `q_alpha` match an independent hand computation to 1e-7 **with the exact frozen kernel conic pairing**; bucket split correct; verdict rule correct; analyzer on true 3-channel.
- `tests/_debug.py`: cross-checks.

The revised kit's algebra, raw-frame adjoint, and native tile×batch bucket walk are all **correct**.

---

## 3. Remote execution outcome (EPIC-05, 8× A100-SXM4-80GB)

### Stage 0 — identity: **PASS (14/14)**
- 10 frozen-source sha256 MATCH (core `935ea6c0…`, c0 ext `d26cccf4…`, fused_ssim `dd6a4c79…`, P12 trainer `675b0267…`, gaussian_model / config / colmap_reader / phase7 / gaussian_inference).
- 2 kernel **byte-identity** BYTE-IDENTICAL (HigsNativeBackward.cu `6efb9d25…`, RasterizeToPixels3DGSDevice.cuh `603d203a…`).
- room/25000 `b31e5a6b…` MATCH; cam19 DSCF4819 MATCH.
- N = 841,781; active_sh_degree = 3.

### Stage 1 — exact-backward algebra: **PASS**
The kernel emits exact `q_alpha` (FP32) and exact 3-channel `q_color`.

### Stage 2 — native graph + raw-frame adjoint: **BLOCKED (OOM)**
- The frozen C0 rasterizer's `intersect_tile` allocates `visN × (n_tiles+1) = 757,119 × 8,161 = 6.18e9 uint32 ≈ 24.7 GiB` plus ~69.15 GiB of per-Gaussian state → **peak ~71 GiB** on the 79.25 GiB A100, leaving ~9.5 GiB free. The 45.86 GiB `intersect_tile` allocation OOMs **deterministically** (all 8 GPUs, co-tenant-independent; retry loop exhausted).
- **Empirical proof:** 720p render **OK** (peak 65.64 GiB); 1080p render **OOM**. The OOM is a frozen-rasterizer memory-capacity limitation, **not** a bug in the revised kit.
- The frozen *training* itself peaked at 1.57 GiB only because `peak_vram_gb` is measured mid-training (post-densification, lower N); the 841K checkpoint is the densest state and is the one P10 targets.

**Consequence:** the raw-frame adjoint, bucket census, Q dump, and oracle SVD never execute. No Q-alpha / Q-color artifacts were produced.

---

## 4. What this does NOT block
- Stage-0 identity (COMPLETE + PASS) — strongest provenance evidence.
- The exact-backward algebra gate (Stage 1) — PASS.
- The revised kit's correctness (synthetically verified; bit-identical to the frozen analyzer + hand computation).

## 5. What would unblock (lead decision — NOT a rescue experiment)
- Run the **exact frozen recipe** on a GPU with **≥ ~120 GiB HBM** (H100-80GB / A100-80GB is insufficient for the 1080p intersect_tile at N=841K; the 841K room is the densest C0 state).
- OR reduce `N_visible` on a host where the 1080p `intersect_tile` fits (would change the frozen culling recipe → **not permitted** by the P10 stop discipline).

**Stop discipline honored:** no full A0-v2, no extra cameras/scenes/checkpoints, no reconstructed VJP, no production CUDA optimization, no rescue experiment. The frozen recipe was preserved throughout.

---

## 6. Artifacts
- **Local:** `A0_P10_PREFLIGHT_KIT/` (this kit), `A0_P10_PREFLIGHT_KIT.zip` (repo root),
  `A0_P10_PREFLIGHT_KIT/remote_artifacts/` = { `A0_SOURCE_IDENTITY.json` (PASS 14/14), `A0_VERDICT.json` (A0P_MEMORY_BLOCKER), `run.log`, `launcher.log` }.
- **Remote (EPIC-05 `/root/A0_P10/out/`):** same 4 files + repro scripts
  `_remote_oomtest.py` (720p OK vs 1080p OOM), `_remote_finalprobe.py` (exact 1080p OOM + per-step memory).

## 7. Packaging note (v2.1)
The v2 zip (40,480 bytes) predates two post-package ops-only additions to `a0p10_run.py` (`--gpu` flag + retry-on-OOM loop). The **deployed remote copy** includes them; the v2 zip does not. This v2.1 summary documents the execution outcome. If the kit is re-shipped, regenerate the zip from the current `a0preflight/` tree (driver + lib + identity + oracle + tests + README) so the `--gpu`/retry ops are captured.
