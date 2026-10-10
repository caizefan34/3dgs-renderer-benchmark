# WV-HIER-R1-MX — Stage A Corrections & Revised-Stage-B Audit Note
**Date:** 2026-10-08 · **Trigger:** received `WV_HIER_R1_MX_STAGE_B_REVISED_20261008.zip`
**SHA256 of received zip:** `2C3AC52954E7233B44B0502B13E670E4B9E8717402D6C18DF275C36BE95C4B55`

---

## 1. What the received package is (and is not)

The bundle is a **revised Stage B-min protocol**, not new measurement results. It contains 4 files (all hash-verified against its own `MANIFEST_SHA256.json`, 4/4 MATCH):

| file | bytes | role |
|---|---|---|
| `README_DSH.md` | 4570 | DSH task instructions, execution gate, decision vocabulary |
| `STAGE_B_REVISED_SPEC.md` | 6650 | corrected scientific protocol |
| `hwcs_mask_oracle.py` | 4519 | GPU-agnostic PyTorch mask oracle (G16/G4/HWCS1/HWCS2) |
| `test_hwcs_mask_oracle.py` | 2811 | CPU-only preflight unit tests |

**It is explicitly `NOT_EXECUTED_ON_MX`** — no MX GPU timings, no CUDA kernel, no checkpoints, no real HWCS NMSE. It is a *stronger protocol than my Stage A/B framing*, and it corrects several over-claims I had made. I accept those corrections.

## 2. Independent audit of the revised oracle (numpy re-implementation)

I do not have torch locally, but `hwcs_mask_oracle.py` is pure array manipulation, so I re-implemented it faithfully in numpy and ran the equivalent of its tests. **Results:**

| Check | Result |
|---|---|
| Shapes / weights ∈ {0, 1/p} on boundary h=65, w=77 (non-multiples of 4/16) | **PASS** |
| **Endpoint identity (bitwise):** `G16 == HWCS(q=p, r=1)` | **PASS** (True) |
| **Endpoint identity (bitwise):** `G4 == HWCS(q=1, r=p)` | **PASS** (True) |
| Per-pixel inclusion frequency = p (unbiasedness), all 4 methods | **PASS** (0.2491–0.2506 vs p=0.25) |
| Exact variance blend with **signed** contributions | **PASS** (rel err 0.06–0.43%) |
| Signed within-parent cross-term `s2−s1` | **−23.8 (negative)** → ordering not universal |

(The oracle is correct; my first audit run showed a false 0.30 inclusion reading due to a normalization bug in *my* test harness — divided by `h*w` instead of by seed-count `N`. After fixing my bug, inclusion = p exactly. The oracle itself was never wrong.)

**Conclusion:** the revised oracle is a *correct and stronger* validation harness than my Stage A independent-seed blend — it uses **common random numbers** (shared `(U_parent, U_child)` per camera/seed) and **bitwise endpoint checks**, which is the right way to test the estimator identity.

## 3. Corrections accepted (all scientifically valid)

| # | My original Stage A claim | Revised-spec correction | Disposition |
|---|---|---|---|
| **C1** | `alpha = r(1−q)/(1−p*q*r)` in JSON identity strings | `1−p*q*r = 1−p²` is wrong; should be `1−p`. My *numerical* alpha (1/3, 1/9) was always computed with `1−p`, so this is a doc-string typo only. | **FIXED** in `stage_a_summary.json` + `STAGE_A_VERDICT.json` (marked CORRECTED, numerics unaffected). |
| **C2** | I treated child-refine efficiency `κ` as identifiable from geometric tile counts and set "κ ≤ 0.5 ⇒ below G4 floor" as a GO gate (my F2). | `κ` is an **operation-cost-efficiency** parameter; raw counts **cannot** identify it. My F2 is invalid as a gate. | **ACCEPTED.** Work-floor undercut + break-even κ\* relabeled **N1-conditional + cost-model-conditional**; F2 removed as a GO gate. |
| **C3** | "G4 ≤ HWCS-2 ≤ HWCS-1 ≤ G16 in 9/9 states" implied as a general property. | Strict ordering is **scene-dependent**: signed child-gradient cross-terms can be negative (s2<s1), flipping it. Only the convex blend is universal. | **ACCEPTED.** Ordering kept for the 9 observed states but labeled scene-dependent; my signed audit independently produced s2−s1=−23.8. |
| **C4** | My Stage A validated the blend with independent Bernoulli draws + aggregate means. | Use **common random numbers** + bitwise endpoint identities for a much stronger test. | **ACCEPTED.** numpy re-audit confirms the CRN oracle passes. |
| **C5** | My Stage A compared only against Native-G4 = **N1** (dense-bin-then-mask). | Must distinguish **N2** (mask-then-bin: sample child tiles, then bin/sort only selected). If the field's "Native G4" is N2, HWCS has **no** structural edge. | **ACCEPTED.** N1/N2 split added; all work-saving claims marked N1-conditional. |
| **C6** | My Stage B spec used the 8-camera aggregate predicted point (1.4247) as a hard 2-camera falsification band. | Use **per-seed paired** differences `d(s)=NMSE_HWCS(s)−[(1−α)NMSE_G4(s)+α·NMSE_G16(s)]` with paired bootstrap CI; report INCONCLUSIVE on wide CI, never auto-DROP. | **ACCEPTED.** Falsification protocol rewritten to per-seed paired CI. |

## 4. What does NOT change

- **Verdict remains `CONDITIONAL_RESEARCH`.** The corrections *tighten* the framing (κ is not count-identifiable; ordering is scene-dependent; N2 baseline missing) but do not alter the fundamental fact that **nothing has been measured on GPU**. No real HWCS NMSE, no CUDA timing, no N1-vs-N2 measurement exist.
- **Stage B-min is still NOT EXECUTED.** It requires an idle MX A100 (not TEMPORAL_COVERAGE-R1), which is unreachable from this host. The revised bundle is the *protocol* to run there, not results.
- The prior WV0 `BLOCKED_RAW_EVIDENCE` verdict is untouched. No old GO restored.

## 5. Net assessment of the revised protocol

It is **methodologically superior** to my Stage A/B framing on every point it raises. I adopt it as the controlling Stage B-min protocol. Its decision vocabulary is well-formed:
`ORACLE_PASS` / `ORACLE_INCONCLUSIVE` / `MODEL_OR_IMPLEMENTATION_MISMATCH` / `STRUCTURAL_POTENTIAL` / `TIMING_NOT_MEASURED` / `CODEX_CUDA_PROTOTYPE_PROPOSAL` — none of which is a performance GO on its own.

The one thing to watch: the revised spec's `STRUCTURAL_POTENTIAL` verdict still rests on the N1 assumption. The counts audit must report **both N1 and N2** Native-G4 baselines before any HWCS dominance claim, exactly as the spec demands (`BASELINE_N1_N2.md`).

## 6. Actions taken on my deliverables

- `stage_a_summary.json`: identity string corrected (C1); ordering caveat (C3); N1-conditioning note (C2/C5); `corrections_accepted` block.
- `STAGE_A_VERDICT.json`: identity string corrected (C1); ordering labeled scene-dependent (C3); `corrections_accepted_20261008` block with the independent numpy-audit result.
- This note.
- Manifest regenerated.
