#!/usr/bin/env python3
"""
WV-HIER-R1-MX — Round 8: quality-gate status under the blend model.

The goal's quality gate is "HWCS NMSE below the G16/G4 mixture". The blend identity
V_HWCS = (1-a)V_G4 + a*V_G16 implies that IF NMSE follows the blend, then HWCS NMSE
equals the mixture EXACTLY (equality, not advantage). This script verifies that on the
Stage A predictions and makes explicit what it means for the quality gate: any advantage
(HWCS below mixture) requires negative within-parent cross-terms (s2<s1), which is
scene-dependent (correction C3) and NOT guaranteed by the blend.
"""
import json, os
import numpy as np
import pandas as pd

OUT = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx"
df = pd.read_csv(f"{OUT}\\stage_a_hwcs_predictions.csv")

rows = []
for _, r in df.iterrows():
    a1, a2 = r.hwcs1_alpha, r.hwcs2_alpha
    mix1 = (1-a1)*r.nmse_G4 + a1*r.nmse_G16     # HWCS-1 mixture
    mix2 = (1-a2)*r.nmse_G4 + a2*r.nmse_G16     # HWCS-2 mixture
    rows.append(dict(state_id=r.state_id,
        nmse_G16=r.nmse_G16, nmse_G4=r.nmse_G4,
        hwcs1_pred=r.hwcs1_nmse, hwcs1_mixture=mix1,
        hwcs1_minus_mixture=r.hwcs1_nmse - mix1,
        hwcs2_pred=r.hwcs2_nmse, hwcs2_mixture=mix2,
        hwcs2_minus_mixture=r.hwcs2_nmse - mix2,
        hwcs1_band=r.hwcs1_band, hwcs2_band=r.hwcs2_band))
out = pd.DataFrame(rows)
out.to_csv(f"{OUT}\\stage_b_quality_gate.csv", index=False)

pd.set_option("display.width", 240, "display.float_format", lambda x: f"{x:.4f}")
print("Quality-gate status: blend-predicted HWCS NMSE vs G16/G4 mixture")
print("="*96)
print(out[["state_id","nmse_G16","nmse_G4","hwcs1_pred","hwcs1_mixture","hwcs1_minus_mixture",
           "hwcs2_pred","hwcs2_mixture","hwcs2_minus_mixture"]].to_string(index=False))

max_dev1 = float(out.hwcs1_minus_mixture.abs().max())
max_dev2 = float(out.hwcs2_minus_mixture.abs().max())
print(f"\nmax |HWCS-1 pred - mixture| = {max_dev1:.2e}  (blend predicts EQUALITY)")
print(f"max |HWCS-2 pred - mixture| = {max_dev2:.2e}")

verdict = {
  "evidence_class": "DERIVED_FROM_STAGE_A_PREDICTIONS",
  "finding": ("The blend model predicts HWCS NMSE EQUALS the G16/G4 mixture (max deviation "
     f"{max(max_dev1,max_dev2):.2e}), NOT below it. The quality gate 'HWCS below mixture' is "
     "therefore NOT satisfied by prediction."),
  "implication": ("Any quality ADVANTAGE (HWCS below mixture) requires negative within-parent "
     "cross-terms (s2<s1), which is scene-dependent (correction C3) and NOT guaranteed. The blend "
     "gives equality; advantage is a possibility to be tested on REAL gradients, not a prediction."),
  "stage_b_min_test": ("Stage B-min must measure REAL HWCS NMSE and compare to the per-seed "
     "G16/G4 mixture (paired CI). HWCS < mixture with CI excluding 0 => scene-dependent advantage "
     "realized. HWCS ≈ mixture => blend confirmed, no quality edge (but cost edge may still exist "
     "vs N1). HWCS > mixture => negative result, DROP quality claim."),
  "quality_gate_status": "NOT_PREDICTED_ADVANTAGE_BLEND_GIVES_EQUALITY",
}
json.dump(verdict, open(f"{OUT}\\stage_b_quality_gate.json","w"), indent=2)
print("\n" + json.dumps(verdict, indent=2))
