# WV-HIER-R1-MX — Stage B Specification
## Minimal GPU Forensic Replay (PyTorch oracle — NO CUDA build)

**Status:** SPECIFIED, NOT EXECUTED. Cannot run on this host (no MX access, no torch, wrong GPU arch, checkpoints on MX). Requires research-lead dispatch to an **idle** MX A100.

---

## 1. Purpose

Resolve the **two decisive unknowns** that Stage A could not, both cheaply and falsifiably:

1. **U1 — Real HWCS gradient quality.** Stage A *predicted* HWCS-1/2 NMSE as a blend of observed G16/G4 NMSE. Measure the actual HWCS gradient NMSE to confirm/refute the blend model.
2. **U2 — Real child-refine cost κ.** The entire GO direction hinges on whether HWCS's structural advantage survives at a favorable κ. Measure the actual per-tile intersection entry counts for G16 / G4 / HWCS.

Stage B-min does **not** measure CUDA kernel/F+B time (no sparse renderer exists) and does **not** claim training impact. It validates the *oracle* and the *structural counts* only.

## 2. Environment gate (hard requirements)

- GPU: **1× idle A100-PCIE-40GB** (MX). **MUST NOT** be the TEMPORAL_COVERAGE-R1 dedicated GPU. Confirm idle via `nvidia-smi` (≤1% util, ~4 MiB) before launch; abort otherwise.
- Env: **WV0 env only** — gsplat 1.5.3, torch 2.4.1+cu124, CUDA 12.4, driver 595.71.05, C++11 ABI=0. **Do not** mix the HiGS cu128 line, do not reinstall/recompile, do not overwrite history.
- Checkpoint: **bicycle__iter15000** (highest-N, N=3,903,754; SHA `a2c82199…37329`).
- Cameras: **2** (IDs 0 and 8 — bicycle cam0/cam8, 540×813).
- Fixed input hashes + seeds recorded; no optimizer, **0 training steps**.

## 3. Experiment matrix

| Axis | Values |
|---|---|
| Granularity | G16 (baseline), Native G4, HWCS-1 (q=.5,r=.5), HWCS-2 (q=.75,r=1/3) |
| Marginal p | 0.25 (all four — matched) |
| Mask seeds | **32 paired** (same seed → same underlying uniform sequence across methods for paired covariance) |
| Dense reference | 2 (one per camera) |
| Cameras | 2 |

**Total:** 4 methods × 2 cams × 32 seeds = **256 sparse gradient evaluations** + 2 dense references. Identical full-render + `1/p` masked-SSE-loss protocol as `_wv0_grad.py` (so results are directly comparable to the 9-state summary).

## 4. Predicted outcomes (falsification bands, from Stage A)

| Predicted at bicycle__iter15000, p=.25 | Band |
|---|---|
| NMSE HWCS-1 | 1.30 – 1.55 (point 1.4247 ± α·½·gap) — DERIVED prediction, not measured |
| NMSE HWCS-2 | 1.15 – 1.33 (point 1.2445 ± α·½·gap) — DERIVED prediction, not measured |
| HWCS-1 child-refine entries | structural count only; κ is NOT count-identifiable — see correction C2 |

> **SUPERSEDED 20261008:** This Stage B spec predates the controlling protocol
> `WV_HIER_R1_MX_STAGE_B_REVISED_20261008`. The F2 row below ("entries > G4 ⇒ DROP") is
> **invalid as a GO/DROP gate** because raw counts do not identify the operation-cost
> parameter κ. Use the revised spec's per-seed paired-bootstrap falsification instead.
> See `STAGE_A_CORRECTIONS.md`.

**Falsifiers:**
- **F1:** measured HWCS-1 NMSE ∉ [1.30,1.55] **or** HWCS-2 ∉ [1.15,1.33] → blend model wrong → **DROP** the current HWCS design (variance model falsified). *(Note: a 2-camera pilot may be underpowered — prefer the revised spec's per-seed paired CI and report INCONCLUSIVE rather than auto-DROP.)*
- **F2:** ~~measured HWCS child-refine entries > Native-G4 child entries (κ ≥ 1, no structural edge) → DROP~~ **SUPERSEDED** — counts do not identify κ; structural counts are reported as COUNTS/PROXY, not as a κ GO gate.
- **F3:** HWCS-1 NMSE ≈ G16 (no quality gain over the cheap baseline) → mechanism useless even if cheap → **DROP**.

## 5. Required saved evidence (per sample, not just summaries)

- Per camera / seed / group: `nmse`, `cosine`, `rel_l2`, dense-grad norm, estimate-grad norm, `dot(dense, estimate)`.
- Sampling mask bits + SHA (reconstructable); per-tile selected/child counts.
- **Exact structural counts:** per-tile intersection entries emitted for G16, G4, HWCS (counted on the actual projection, not the floor/closed bbox proxy) → this is the empirical κ.
- Fixed-camera online **mean & variance (M2) accumulators** for paired covariance across seeds (state retention policy in manifest).
- All input SHA (checkpoint, cameras, images), source SHA, `.so` SHA, seed base, torch/CUDA/driver versions.

**Prohibited:** saving only summaries (the original WV0 failure mode); calling full-render+masked-loss timing "sparse CUDA F+B"; using the same pixel budget as a cost proxy.

## 6. Code patch (HWCS two-stage mask)

Add to `_wv0_grad.py` (mirror of the existing `tile_pixel_mask_fast`); the rest of the pipeline is unchanged:

```python
def hier_child_pixel_mask_fast(H, W, q, r, seed, device="cuda"):
    """HWCS two-stage mask. Parent=16x16, child=4x4 (16 children/parent).
    Parent active w.p. q; child active w.p. r given parent active.
    Marginal child inclusion p=q*r; HT weight 1/(q*r). Returns pixel mask,
    child-inclusion count, parent-active count, and (parent,child) tile ids."""
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    PG = 16; CG = 4
    pth, ptw = math.ceil(H/PG), math.ceil(W/PG)      # parent tiles
    CPG = PG // CG                                    # children per parent axis = 4
    cth, ctw = math.ceil(H/CG), math.ceil(W/CG)        # child tiles
    # stage 1: parent activation
    par_active = (torch.rand(pth*ptw, generator=g) < q).float()
    # stage 2: child activation (only meaningful where parent active)
    child_active = (torch.rand(cth*ctw, generator=g) < r).float()
    # child marginal indicator = par_active(parent(child)) * child_active
    par_of_child = (torch.arange(cth).div(CPG, rounding_mode="floor").clamp(max=pth-1)[:,None]*ptw
                  + torch.arange(ctw).div(CPG, rounding_mode="floor").clamp(max=ptw-1)[None,:])
    xi = par_active.reshape(-1)[par_of_child.reshape(-1)] * child_active   # (cth*ctw,)
    # build pixel mask from child ids
    ci = torch.arange(cth).div(CPG, rounding_mode="floor").clamp(max=pth-1)[:,None].expand(cth,ctw).reshape(-1)
    cj = torch.arange(ctw).div(CPG, rounding_mode="floor").clamp(max=ptw-1)[None,:].expand(cth,ctw).reshape(-1)
    child_id = (torch.arange(H).div(CG, rounding_mode="floor").clamp(max=cth-1)[:,None]*ctw
              + torch.arange(W).div(CG, rounding_mode="floor").clamp(max=ctw-1)[None,:])
    child_id = child_id.reshape(-1)
    m = xi.reshape(-1)[child_id] > 0.5
    p = q*r
    w = torch.where(m, torch.tensor(1.0/p, device="cuda"), torch.tensor(0.0, device="cuda"))
    return w.to(device), int(xi.sum()), int(par_active.sum()), (cth*ctw)
```
Then a `sparse_hier_grad(...)` mirroring `sparse_tile_grad` but using `hier_child_pixel_mask_fast` and reporting the child-intersection entry count `xi.sum()` and parent-active count for κ. **Validation gate before the run:** on a 64×64 toy, assert (a) per-pixel inclusion frequency ≈ p over many seeds, (b) empirical Var(ĝ) matches the §4 analytic blend within MC error, (c) `E[ĝ]=g` (unbiased) — reuse `stage_a_variance.py` as the reference.

## 7. Cost estimate & scale guard

- 256 sparse + 2 dense full-renders at N=3.9M, 540×813: **≈10–30 min** on one idle A100 (original WV0: ~2–5 s/state-cam-eval full-render; 258 evals ≈ 10–20 min).
- Peak VRAM: checkpoint 968 MiB + autograd graph for ~3.9M Gaussians × {means,quats,scales,opacities,SH3 colors} + render buffers ≈ **3–5 GiB** — fits 40 GiB with large margin.
- **Scale freeze:** do NOT expand to 9 states / 18,432 estimates / 3K-30K training without a separate authorization. This is a provenance + repeatability + κ pilot, **not** a replacement for the original 9-state claim.

## 8. Decision rule after Stage B-min

| Outcome | Action |
|---|---|
| F1 triggered (blend wrong) | **DROP_HWCS_CURRENT_DESIGN** |
| F2 triggered (κ≥1, no structural edge) | **DROP** or redesign parent-reuse |
| F3 triggered (no gain over G16) | **DROP** |
| All bands pass AND κ ≤ ~0.5 AND HWCS-1 below G4 floor in real counts | → authorize **Stage B-cuda** (minimal sparse CUDA prototype) feasibility review = path to **CODEX_FEASIBILITY_GO** |
| Bands pass but κ ∈ (0.5, 0.75] | **CONDITIONAL_RESEARCH** — HWCS below mixture but near break-even; needs CUDA timing to decide |
