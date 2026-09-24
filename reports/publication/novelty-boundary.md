# Novelty-Boundary Audit

**Report ID:** publication-novelty-boundary
**Date:** 2026-09-24
**Constraint:** internal evidence only. No new experiments, no literature search performed here.
**Rule:** every literature-dependent statement carries the literal tag `LITERATURE_VERIFICATION_REQUIRED` and must not be asserted as fact in the paper until verified.

---

## 1. What can be claimed safely from internal evidence

### 1.1 Likely core contribution — *exact work elimination under unchanged semantics*
- **Claim shape:** the projected-state materialization in the forward path can be eliminated **exactly** — same loss, same gradients, same optimization trajectory semantics — with a measured forward-stage gain.
- **Evidence:** E05 (F9 exactness/oracle gates + standalone timing: forward +18.9 / +14.8 / +23.2%), and the *flat forward phase* in the full-training result (4.507 -> 4.523 ms) which demonstrates that the forward path was not perturbed at 30K.
- **Strength:** strongest internal footing in the paper. Exactness is proved, not argued.
- **Boundary:** this is an elimination of *redundant materialization*, not a change to the rasterizer's gradient topology. It does not touch the pixel recurrence.

### 1.2 Supporting engineering contribution — *scalar adjoint simplification*
- **Claim shape:** the vector adjoint carried alongside the projection can be collapsed to its scalar-sufficient form, removing redundant adjoint arithmetic exactly.
- **Evidence:** E07 — exactness closure (`h2-bwd-2r-exactness-closure.md`), freeze smoke, and the C0 nested contribution (1.2 / 0.1 / 0.3%).
- **Strength:** exactness is established; magnitude in the composed stack is **small**.
- **Boundary:** must be presented as supporting engineering. Claiming it as the core contribution overstates the measured effect.

### 1.3 Candidate scientific insight — *sufficient-statistic differentiation (moment-space geometry adjoint, H8-MR)*
- **Claim shape:** geometry differentiation can be reformulated in the space of sufficient statistics (moments) rather than per-primitive materialized state, giving an exact geometry adjoint with fewer redundant operations.
- **Evidence:** E06 — moment-adjoint gates (`h8-0-moment-adjoint-gate.md`, `h8-0r-opacity-absorbed-moment.md`, `h8-mr-production.md`) + direct backward timing (4.31 / 4.11 / 3.54%, geomean 3.97%; F+B geomean 2.09%).
- **Boundary:** no causal attribution of H8 inside the composed C0 stack is authorized (registry SS35). The framing is a **candidate** distinction.
- **Tag:** `LITERATURE_VERIFICATION_REQUIRED` — whether sufficient-statistic moment-space geometry adjoints are previously described is not established by internal evidence.

### 1.4 Negative / mechanistic insight — *why the obvious hierarchical route does not pay*
- **Claim shape:** hierarchical reuse of geometry work is real, but the reusable fraction is largely *already removed* by ordinary last-id pruning, so hierarchical backward acceleration is not profitable on the final stack.
- **Evidence:** E08 (P2-1A gates: B_support FAIL, D_per_splat_weight FAIL, E_rgb_alpha FAIL, `timing_authorized: false`), E09 (P2-1C: entry-weighted 10.8 / 4.7 / 8.2% but **100% overlap** with last-id pruning; frozen-128G skippable only 0.40 / 2.00 / 1.08%), E10 (P3-H: `HAR_CEILING_WEAK`, ALL_ACCUM_CEILING ≤ 0.54% backward).
- **Strength:** a clean, quantified negative result. It is the paper's §4.
- **Boundary:** DIAGNOSTIC_ONLY, 3-scene renderer fixtures. Appendix-grade numbers; the *insight* is main-paper material, the *timings* are not.

---

## 2. Known prior-art-like components (internal classification)

From `artifacts/higs-p3-0/prior_art_overlap.json`:

| Component | Internal classification | Distinction the audit can state without literature claims |
|---|---|---|
| Per-Gaussian backward | **PARTIAL_OVERLAP** | It changes *recurrence ownership*; C0 V3 proves the pixel recurrence is **untouched**. |
| HAR P_GEOM packet | overlaps **the H8 moment set** | The H8 moment set is the same object; H8-MR's distinction is that it is used as an *exact differentiable* geometry adjoint, not as a caching packet. |
| Kernel-fusion-style materialization removal (F9) | engineering, prior-art-like | F9's only defensible distinction is *exactness* (gradient/loss equivalence), not the act of fusion. |

`prior_art_overlap.json` itself records `candidate_scientific_distinction_status: "STATED, NOT CLAIMED"`. That status must be preserved in the paper.
`artifacts/research-registry/prohibited_claims.json` forbids prior-art novelty claims presented as fact.

---

## 3. Test of the candidate framing

> **Exact Work Elimination + Sufficient-Statistic Differentiation for Differentiable Gaussian Splatting Training**

| Module | Fits the framing? | Role in the framing | Honest limitation |
|---|---|---|---|
| **F9** | Yes — the *exact work elimination* half | Eliminates projected-state materialization exactly | "Elimination of redundant materialization" is an engineering pattern, not a new theory; its scientific content lies in proving semantics are unchanged |
| **Scalar Adjoint** | Yes — but weakly | A second instance of the same principle applied to the adjoint (carry the sufficient scalar, not the redundant vector) | Algebraic simplification; magnitude in C0 is 0.1–1.2% |
| **H8-MR** | Yes — this is where *sufficient-statistic differentiation* is actually a claim | Reformulates geometry differentiation in moment space; the only module that can carry the scientific thesis | No causal attribution in the composed stack; novelty unverified |

**Assessment:** the framing is **coherent but currently under-substantiated**. F9 and Scalar naturally exemplify "exact work elimination"; H8-MR is the only module that supports "sufficient-statistic differentiation". The framing is therefore best written as a *hypothesis that organizes the three transforms*, substantiated by the cumulative ablation — which is exactly the evidence that does not yet exist at 30K (R-12). Until A1/A2 exist, the framing must be labelled a framing hypothesis, not a demonstrated theory.

**Do not write:** "we introduce sufficient-statistic differentiation" (as an established concept) — `LITERATURE_VERIFICATION_REQUIRED`.

---

## 4. Statements requiring external verification

```text
LITERATURE_VERIFICATION_REQUIRED
  - that no prior work performs EXACT projected-state elimination while preserving the pixel recurrence
  - that sufficient-statistic / moment-space geometry adjoints for Gaussian splat training are not previously described
  - that scalar-adjoint collapse of the projection VJP is not already standard in gsplat-family trainers
  - that per-Gaussian backward / recurrence-reassignment work (Faster-GS, Speedy-Splat lineage) is distinguishable
    from the reclaimed-work claim made by C0 V3
  - any statement of the form "first", "novel", "unlike prior work"
```

---

## 5. Bottom line

- **Safe to claim today:** exactness of the three transforms; the flat forward phase under full training; the measured backward-stage decrease; the negative result that hierarchical reuse alone does not pay.
- **Safe only as a labelled hypothesis:** the unifying "exact work elimination + sufficient-statistic differentiation" framing.
- **Not safe to claim:** any novelty/primacy statement, and any per-module share of the 1.0685x.