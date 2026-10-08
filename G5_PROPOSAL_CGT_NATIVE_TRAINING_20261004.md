# G5 Proposal – CGT as a Trainable Scene Representation (Native CGT Training)

- **Stage**: G5 (proposal only – NOT implemented). Predecessor: CGT-F0 Fused CUDA Kill-Shot = `PASS_STRONG_GPU_TRANSLATION` (pooled S_mixed = 2.99 – see `cgt_f0_fused_cuda/results/F0_VERDICT.json`).
- **Date**: 2026-10-04. **Status**: PROPOSAL / pre-registration draft. No G5 code, kernel, training loop, or backward path exists or is implied by this document.
- **Scope boundary inherited from F0**: F0 delivered a *forward-only* fused renderer (student + rigorous certificate + canonical fallback, one kernel). G5 is a distinct, larger stage: make the CGT coefficients themselves trainable scene primitives and build the CGT *backward* path. F0 explicitly excluded training/backward/densification (goal §22, §23).

---

## 1. Motivation (evidence-linked, not assumed)
- F0 measured a genuinely fused forward CGT render path: **S_mixed = 2.99** pooled (median-ratio; mean-ratio 3.03), cross-GPU/run 2.88–2.99 (~5% spread), correct to ≤1.3e-6 fp32 (gate 2e-5, ≥15x margin), with rigorous-safe semantics and ~free dispatch (weighted fallback 0.175, safe pass 0.827).
- Translation efficiency **eta_translation_safe = 0.367**: the GPU realizes ~37% of the intrinsic safe-path speedup; S_mixed realizes ~3.0x vs the 5.46x actual work-reduction. The compression survives GPU translation but is partly eaten by fixed per-tile overhead and the memory-bound teacher.
- Per-scene is **workload-dependent**: room 3.01 / garden 3.28 / bicycle 1.54. bicycle (light canonical teacher, 0.184 ms) is capped by the fused path ~0.12 ms fixed cost – it is the **least** accelerated and therefore the regime with the most remaining headroom.
- **Conclusion driving G5**: a 2.99x quality-preserving *renderer* speedup clears the updated promotion bar (≥2.0x – `PASS_STRONG_GPU_TRANSLATION`) and justifies designing the next stage. It does **not** by itself establish a training speedup (that requires G5).

## 2. Core scientific question (G5)
> Can the CGT compressed representation become a **trainable** scene primitive directly – i.e., can we train the Gaussian scene *through* the rank-R CGT coefficients (student parameters +, if admitted, the teacher), eliminating the repeated per-frame teacher→student fitting step and its host-side cost – while preserving scene quality (PSNR/SSIM) and convergence behavior?

Sub-questions:
- **Q5.1 Backward existence/quality**: does a correct, stable CGT backward (dL/d(coeffs) via the existing `predict_and_vjp` VJP) exist and match autograd through the fused forward to a tight tolerance?
- **Q5.2 End-to-end benefit**: with a CGT backward in place, what is the projected and measured training-level speedup (Amdahl over the full iteration)? F0 could only *project* this (f = renderer fraction of training was ASSUMED 0.4–0.6, NOT_MEASURED).
- **Q5.3 Coefficient plasticity**: do the rank-R CGT coefficients, trained directly, retain fidelity to the full Gaussian scene across the convergence trajectory (no per-iter refit), or does drift accumulate (the failure mode seen in the sparse-update/importance lanes)?
- **Q5.4 Regime transfer**: does the benefit concentrate in the light-workload (bicycle-like) regime where F0 left the most headroom, and does it hold for heavy-Gaussian scenes?

## 3. Prerequisite measurements (must be done BEFORE any G5 implementation)
These are the gaps F0 explicitly left open; G5 should not start until they are measured:
- **M5.1 Renderer fraction f (training)**: instrument one real 3DGS training iteration (e.g., the frozen B1 baselines used upstream) to measure the true forward-render fraction of iteration time. F0 used an ASSUMED f = 0.4–0.6; the Amdahl projection (S_train ~1.37–1.67x) is only as good as this input.
- **M5.2 CGT backward cost**: measure a VJP/backward through the fused path (the `predict_and_vjp` already provides the math; the kernel does not yet). Report dL/dcoeffs time vs the canonical backward.
- **M5.3 Refit amortization baseline**: quantify the current per-scene teacher→student fitting cost that G5 aims to eliminate (host + device), so the "eliminate refit" claim is measured, not asserted.
- **M5.4 Nsight hardware counters on the F0 fused kernel** (optional but recommended): F0 marked these NOT_MEASURED; a real occupancy/DRAM/L2 profile would confirm whether the memory-bound teacher or fixed launch overhead dominates the residual 63% of intrinsic gamma.

## 4. Proposed G5 method (design sketch – to be pre-registered, not yet validated)
- **Representation**: keep the frozen-Gaussian scene as the *initialization* source; promote the rank-R CGT coefficients (per tile/group) to trainable parameters. Teacher becomes an optional periodic anchor, not a per-frame requirement.
- **Forward**: reuse the F0 fused kernel (student fast path + rigorous certificate + canonical fallback). No new forward math.
- **Backward**: implement the CGT VJP (`predict_and_vjp`, reused verbatim from upstream g1) as a fused backward kernel matching the forward memory layout; certify the backward against autograd through the fused forward (tolerance to be pre-registered).
- **Training loop**: standard 3DGS Adam/densification *over CGT coefficients* where a coefficient is trainable; fall back to canonical Gaussian parameters where the certificate flags unsafe. Densification policy over coefficients is an open design choice (see R5.3).
- **Fallback semantics**: the rigorous certificate (G2, GATE=0.02) carries over unchanged – unsafe tiles/regions still render canonically, so quality is bounded by the same guarantee F0 relied on.

## 5. Suggested gates (pre-registration placeholders – NOT final)
- **G5-A Backward correctness**: fused CGT backward vs autograd-through-fused-forward, max-abs error ≤ pre-registered tol (e.g. 1e-4 rel / 2e-5 abs on RGB-relevant adjoints); cosine similarity of parameter gradients ≥ 0.999.
- **G5-B Quality preservation**: trained-to-convergence PSNR/SSIM vs the full-Gaussian baseline within a pre-registered dB margin, on room/garden/bicycle (same cohort as F0/G1–G2). No per-iter teacher refit allowed to "rescue" quality.
- **G5-C Economic speedup**: measured end-to-end training speedup ≥ a pre-registered bar (e.g. ≥ 1.3x) AND/OR a measured reduction in per-iteration host–device refit work ≥ a pre-registered fraction. Both the refit-elimination and the wall-clock must be reported.
- **G5-D No quality laundering**: certificate/fallback rates, memory overhead, and integration complexity reported alongside speed; a speedup that requires disabling the rigorous certificate fails the gate.

## 6. Risks / failure modes (anticipated)
- **Plasticity/drift (Q5.3)**: directly training low-rank coefficients may lose expressivity as the scene updates – the same failure class that sank the importance-guided sparse-update lanes (real degradation >> model). Mitigation: periodic teacher anchoring, but that reintroduces refit cost – must be measured (M5.3).
- **Amdahl ceiling**: if the true renderer fraction f is small (M5.1), even a large renderer speedup yields modest end-to-end gain; G5 could land below the economic bar despite a correct backward.
- **Backward kernel complexity**: a fused CGT backward is substantially harder than the forward; correctness (Q5.1) is a hard prerequisite and a fail-closed stop if it cannot be met.
- **Regime dependence**: benefit may not transfer from heavy-Gaussian (room/garden) to light (bicycle) tiles or vice-versa.

## 7. Relationship to F0 and to the updated gate
- F0 answered its one question cleanly: the fused forward CGT path delivers **S_mixed = 2.99** (quality-preserving, rigorous fallback), which under the **updated** promotion tiers (≥ 2.0x – `PASS_STRONG_GPU_TRANSLATION`) clears the bar for G5 design. Note: the *stale* goal-file text (goal §19/§28) still lists 2.5x/3.0x hard thresholds under which 2.99 would map to `PASS_TO_G5`; the user-provided GATE UPDATE explicitly supersedes those (">=3x is exceptional, NOT required"), and `F0_VERDICT.json` implements the updated tiers. This proposal follows the updated gate.
- **Nothing here is a training speedup claim.** All training-level numbers remain projected/assumed until M5.1–M5.3 are measured. G5 must earn its own verdict.

## 8. Deliverables this proposal implies (when/if authorized)
- A pre-registered G5 execution pack (source lock, gates, cohort, tolerances) – analogous to the F0 pack – separate from `cgt_f0_fused_cuda/`.
- Do **not** begin implementation until the user authorizes a new pre-registered G5 lane (consistent with the knowledge-base todo: "G5 proposal 待单独建；仅当授权新 lane").

## Reproduction / provenance
- F0 evidence: `cgt_f0_fused_cuda/results/F0_VERDICT.json`, `F0_MIXED_TIMING.json`, `F0_AMDAHL_PROJECTION.json`, `F0_REPORT.md`, `F0_CORRECTNESS.json`, `F0_WORK_REDUCTION.json`, `MANIFEST.sha256` (56 entries, MX+local verified).
- Upstream frozen math (read-only, reused): `g1_compressibility.py` (tile_teacher/predict_and_vjp/fit_cgt), `g2_rigorous_certificate.py` (certify_tile/certify_box).
