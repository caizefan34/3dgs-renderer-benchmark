"""Apply the revised-spec corrections to Stage A deliverables.
Corrections (all from WV_HIER_R1_MX_STAGE_B_REVISED, independently audited):
  C1: alpha identity denominator 1-p*q*r (=1-p^2) -> 1-p  (doc-string typo; numerics always used 1-p)
  C2: kappa is an operation-cost-efficiency parameter, NOT identifiable from tile counts;
      the work-floor undercut and break-even kappa* are CONDITIONAL on an assumed cost model
      and on Native-G4 using the N1 (dense-bin-then-mask) baseline.
  C3: variance ordering G4<=HWCS<=G16 holds only when within-parent signed cross-terms are
      non-negative (s2>=s1); it is scene-dependent, NOT universal. The blend identity IS universal.
  C4: stronger validation uses common random numbers (CRN) + bitwise endpoint checks
      (G16==HWCS(q=p,r=1), G4==HWCS(q=1,r=p)); independently re-audited in numpy -> PASS.
"""
import json, os

OUT = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx"

# --- fix summary JSON identity string + add corrections block ---
sp = os.path.join(OUT, "stage_a_summary.json")
s = json.load(open(sp, encoding="utf-8"))
s["variance_model"]["identity"] = ("Var_HWCS = (1-alpha)*Var_nativeG4 + alpha*Var_G16, "
                                   "alpha = r*(1-q)/(1-p)  [CORRECTED: was 1-p*q*r=1-p^2; "
                                   "numerical alpha always used 1-p, MC-verified]")
s["variance_model"]["ordering_caveat"] = ("G4<=HWCS<=G16 holds ONLY when within-parent signed "
                                          "cross-terms are non-negative (s2>=s1); scene-dependent, "
                                          "NOT universal. The convex-blend identity IS universal. "
                                          "Independently confirmed: signed test gave s2-s1=-23.8 (negative).")
s["work_model_N1_conditioning"] = ("The work-floor undercut (HWCS-1 below Native-G4 dense-binning "
                                   "floor) and break-even kappa* are CONDITIONAL on (a) the assumed "
                                   "cost model and (b) Native-G4 using the N1 dense-bin-then-mask "
                                   "baseline. kappa is an operation-cost-efficiency parameter and "
                                   "CANNOT be identified from geometric tile counts; it requires "
                                   "Stage C CUDA timing. If Native-G4 uses N2 (mask-then-bin), "
                                   "HWCS has no structural edge.")
s["corrections_accepted"] = {
   "source": "WV_HIER_R1_MX_STAGE_B_REVISED_20261008 (independently audited, hashes verified)",
   "C1_alpha_denominator_typo": "FIXED (doc string; numerics unaffected)",
   "C2_kappa_not_count_identifiable": "ACCEPTED - work-floor result relabeled N1-conditional + cost-model-conditional",
   "C3_ordering_scene_dependent": "ACCEPTED - caveat added; blend identity remains universal",
   "C4_crn_bitwise_validation": "ACCEPTED - numpy re-audit PASS (endpoints bitwise, inclusion=p, blend exact)",
   "N1_N2_baseline_split": "ADDED - prior Stage A only considered N1",
   "crn_falsification": "ADDED - per-seed paired bootstrap CI replaces hard 8-camera band",
}
json.dump(s, open(sp, "w", encoding="utf-8"), indent=2)
print("updated", sp)

# --- fix verdict JSON identity string + add corrections ---
vp = os.path.join(OUT, "STAGE_A_VERDICT.json")
v = json.load(open(vp, encoding="utf-8"))
v["key_findings"]["variance_identity"] = ("Var_HWCS = (1-alpha)*Var_G4 + alpha*Var_G16, "
                                          "alpha = r*(1-q)/(1-p); HWCS-1 alpha=1/3, HWCS-2 alpha=1/9  "
                                          "[CORRECTED: denominator was mis-typed 1-p*q*r=1-p^2; "
                                          "numerical alpha always used 1-p, MC-verified]")
v["key_findings"]["nmse_ordering"] = ("G4 <= HWCS-2 <= HWCS-1 <= G16 in 9/9 OBSERVED states "
                                      "(valid because within-parent cross-terms were non-negative there); "
                                      "scene-dependent, NOT universal - blend identity is the universal part")
v["corrections_accepted_20261008"] = {
   "source_bundle": "WV_HIER_R1_MX_STAGE_B_REVISED_20261008.zip (SHA256 2C3AC529...C4B55, 4/4 files verified)",
   "independently_re_audited": True,
   "numpy_oracle_audit": "endpoint identities bitwise PASS; per-pixel inclusion=p (0.249-0.251); variance blend exact (0.06-0.43%) with SIGNED contributions (s2-s1=-23.8)",
   "C1_alpha_denominator_typo": "FIXED",
   "C2_kappa_not_count_identifiable": "ACCEPTED",
   "C3_ordering_scene_dependent": "ACCEPTED",
   "C4_crn_bitwise_validation": "ACCEPTED",
   "N1_N2_baseline_split": "ADDED",
   "note": "Corrections tighten framing; they do NOT change the CONDITIONAL_RESEARCH verdict (nothing measured on GPU)."
}
json.dump(v, open(vp, "w", encoding="utf-8"), indent=2)
print("updated", vp)
print("DONE")
