# BES0 — BES-GS Bounded-Event Gaussian Splatting: Execution Report & Verdict

**Protocol:** BES0-v1 (frozen)  **Date:** 2026-10-05  **Host:** MX bms-39468022-001 (8x A100-PCIE-40GB SM80)
**Method:** BES-GS = per-occurrence alpha-bound routing (R/D) + log-hazard majorant jump + conditional-integration D + three-query trainable path.
**Headline verdict:** `NO_GO_OPPORTUNITY` (protocol gate 4). BES-GS's defining mechanism is **structurally empty** on real 3DGS.

---

## 1. Verdict

`NO_GO_OPPORTUNITY` — the E1 investment gate (protocol GATES.md gate 4: "E1 room+other two scenes have no optimistic 3x opportunity OR E2 full-step < 3x") is satisfied by E1 alone. The bounded-event **R** set, which BES-GS routes through a cheap log-hazard jump, contains **rho_R = 1.044e-9** of occurrences on the three frozen scenes (1 routable occurrence out of 957,816,964). With R empty, BES-GS degenerates to a full D-scan (= the native alpha composite) plus bound/prefix overhead, so no speedup is possible; the only real headroom is the per-pixel transmittance early-stop already in gsplat (~1–2%).

E2/E3 were therefore **not run** — this is the protocol-intended stop at gate 4, not an incomplete run.

---

## 2. What was executed (faithful to the frozen spec)

| Stage | Status | Evidence |
|---|---|---|
| **E0** math reference | **PASS** | 11/11 identities (color_expectation, gradient_expectation, analytic_bounds, first_hit_distribution, candidate_count, trial_tail, suffix, wrong_hazard_ratio_rejected, …); `color_maxabs=1.1e-16`, `gradient_maxabs=5.5e-17` |
| **E1** production correctness + full-scene opportunity screen | **PASS** | `raw/E1/opportunity_{room,bicycle,garden}.json` (fresh re-run), `rerun_master.log`, `probe_opportunity.py`; production gsplat rasterizer load+render confirmed |
| **E2** full-step + convergence | **NOT_RUN** | gate-4 short-circuit (no optimistic 3x at E1) |
| **E3** fixed-N pilot (3 scenes x 3 seeds) | **NOT_RUN** | gate-4 short-circuit |
| render eval (spp sweep) | **NOT_RUN** | training-only; no 3x training claim to evaluate |

All MX runs used the fail-closed guard (`mx_guard`) on an idle GPU (<=512 MiB, <=2%, 0 procs, 20 s window). All 8 GPUs were unlocked per the operator amendment ("unlock all"); the idle gate — the timing-integrity control — was unchanged.

---

## 3. Decisive result: the R set is structurally empty

BES-GS routes an occurrence to the cheap **R** path iff its per-tile alpha lower bound `l_i > 0` and `hi_i/l_i <= gamma` (gamma=2). On the three frozen MipNeRF360 scenes (12 dev cameras each, tile size 16 px, protocol tile size):

| scene | visible gaussians | occurrences (ts=16) | n_R (ts=16) | rho_R (ts=16) | rho_R (ts=2) | per-pixel early-stop bound |
|---|---:|---:|---:|---:|---:|---:|
| room    |  9,501,967 | 168,700,377 | 0 | 0.0       | 1.09e-7 | 1.008x |
| bicycle | 34,154,646 | 540,816,678 | 1 | 1.85e-9   | 1.04e-7 | 1.009x |
| garden  | 21,359,326 | 248,299,909 | 0 | 0.0       | 9.94e-8 | 1.020x |
| **total** | **65,015,939** | **957,816,964** | **1** | **1.04e-9** | | **max 1.020x** |

- **gamma-invariant:** rho_R = 0.0 for gamma in {2, 5, 10, 30, 100} at ts=16 (every scene).
- **tile-invariant:** rho_R <= 1.1e-7 even at ts=2 px (1 routable occurrence per ~10M).
- The **optimistic 3x ablation** (NUMERICS.md E1 step 4: "R-events/score all free", keep projection/sort/bounds/prefix/D/optimizer) therefore collapses to the per-pixel early-termination headroom = **1.008–1.020x < 3x** on all three scenes.

### Root cause (why R is empty)
Real 3DGS gaussians are **anisotropic thin splats with a sub-pixel minor axis**. Diagnostic (room cam 1): conic = [9.93, -0.87, 0.076] -> sigma_short ~ 0.32 px, sigma_long ~ 115 px; screen-radius median 93 px. Within any tile >= 2 px, the sub-pixel minor axis drives the per-tile alpha lower bound `l_i = min_{p in tile} alpha_i(p)` to 0 (alpha underflows below the 1/255 cutoff at the tile's far corner). So `is_R = (l_i > 0) & (hi_i/l_i <= gamma)` is **False for essentially every occurrence**, for every gamma and every tile size. The condition is violated structurally, not by parameter tuning.

This is exactly the failure mode METHOD.md §5 lists ("a large fraction of occurrences have l=0") and the SOURCES.md prior art (majorant/null-collision routing is known; the per-occurrence ratio-limited variant has no substrate on thin splats).

---

## 4. Production pipeline correction

The prior report stated the gsplat 1.5.3 CUDA rasterizer was "unloadable on MX (JIT vs torch signature mismatch)". This was **incorrect**: the failure was `FileNotFoundError` from `tempfile.gettempdir()` because the root disk `/dev/sda1` was 100% full — not a JIT mismatch. With `TMPDIR=/dev/shm/bes_tmp` (and `TORCH_EXTENSIONS_DIR` on the storage pool), the production rasterizer **loads and renders** (`render shape (1, 64, 64, 3)`, CUDA extension built in 69 s). The opportunity-screen alpha is gsplat-convention (projection verified to native `means2d` to 6.1e-5 px; alpha fp32-vs-fp64 6.6e-8), so the R-fraction is faithful to production semantics. The R-fraction is a mathematical property of the alpha bounds and is independent of whether the projection is run in torch or in the native kernel.

---

## 5. Gate analysis and reducer note

- **Gate 1 (INCONCLUSIVE_EVIDENCE):** not triggered — E0 PASS, E1 PASS with measurements + source_patch + guard + run_status.
- **Gate 2 (NO_GO_CORRECTNESS):** not triggered — E1 `bound_violations=0`, wrapper parity 6.6e-8 < 1e-5.
- **Gate 4 (NO_GO_OPPORTUNITY):** **triggered** — E1 optimistic speedup < 3x on all three scenes.
- **Reducer note:** the automated `reduce_verdict.py` requires all four stages' evidence and would emit `INCONCLUSIVE_EVIDENCE` because E2/E3 stage JSONs are absent. That is a structural artifact of the reducer's all-stages requirement, which does not model the protocol's gate-4 short-circuit ("E1 no optimistic 3x **OR** E2 step < 3x"). The honest, protocol-faithful verdict is `NO_GO_OPPORTUNITY`, recorded in `VERDICT.json`.

---

## 6. Reproduction

```bash
# on MX, nerficg env, TMPDIR set (root disk full):
export TMPDIR=/dev/shm/bes_tmp TORCH_EXTENSIONS_DIR=/mnt/storage_pool/liaoyuanjun/bes0_mx/torch_ext
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118
export BES0_PACKAGE_ROOT=/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH
# production rasterizer check + full-scene opportunity screen:
bash /mnt/storage_pool/liaoyuanjun/bes0_mx/bes_opp_probe/rerun_opp.sh
# -> /dev/shm/bes_opp/out/opportunity_{room,bicycle,garden}.json ; persisted copy in bes_opp_probe/out/
```
Probe: `bes_opp_probe/probe_opportunity.py` (faithful BES0 `build_bounds` R0 enumeration; tile+gamma sweep; per-pixel early-termination). Guard: `BES0_MX_DSH/scripts/mx_guard.py` (fail-closed, idle-gated).

---

## 7. Deliverables & limitations

- **Deliverables:** this report, `VERDICT.json` (NO_GO_OPPORTUNITY), `E0.json`/`E1.json`/`E2.json`/`E3.json`, `selected_config.json` (FROZEN, gamma=2, tile=16, 3 scenes, seeds 17/29/43, fixed-N, no densification), `raw/E0/` + `raw/E1/` (guard, run_status, measurements, source_patch), `RESULT_MANIFEST.sha256`.
- **Limitations:** E2/E3 not run (gate-4 short-circuit); render eval NOT_RUN (training-only); reducer verifies evidence shape/hashes, not measurement honesty; candidate novelty and disruptive impact unproven (BES-GS is a known combination of majorant routing + log-hazard jump + conditional integration).
- **Conclusion:** BES-GS's >=3x same-quality training-speedup claim is **not supported** on real 3DGS. The bounded-event per-occurrence routing that defines the method is structurally empty (sub-pixel thin splats -> l_i = 0 -> R empty). The only real rendering headroom is the per-pixel transmittance early-stop already present in gsplat (~1–2% aggregate). This is a decisive negative result for the BES0 research direction.
