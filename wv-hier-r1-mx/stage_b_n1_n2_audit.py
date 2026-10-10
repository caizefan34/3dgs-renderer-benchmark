#!/usr/bin/env python3
"""
WV-HIER-R1-MX — N1 vs N2 Native-G4 structural work audit.
Controlling requirement: report BOTH baselines; do not claim HWCS dominance off N1 alone.

Baselines (work in units of I_G16 dense parent intersections; GAM=raster weight=1):
  N1  dense-bin-then-mask : bin+sort ALL I_G4 child entries, then mask.
                            W_N1 = rho + GAM*p*rho = rho*(1 + GAM*p)
  N2  mask-then-bin       : choose child tiles (mask at p), bin+sort ONLY selected.
                            W_N2 = p*rho (bin) + p*rho (sort) + GAM*p*rho (raster)
                            W_N2 = p*rho*(2 + GAM)
  HWCS(κ)                 : parent bin+sort (1) + child refine [p*rho + κ(q-p)*rho]
                            + raster of selected children (GAM*p*rho)
                            W_HWCS = 1 + [p + κ(q-p)]*rho + GAM*p*rho

Key question: does HWCS beat N1? does HWCS beat N2? (N2 is the strong sparse baseline.)
All values are STRUCTURAL INTERSECTION-COUNT PROXIES, NOT measured GPU time.
"""
import json
import numpy as np
import pandas as pd

EV = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx\evidence"
OUT = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx"
cost = pd.read_csv(f"{EV}\\WV0_ABSOLUTE_COST.csv")

def dense_proxy(st, G):
    return float(cost[(cost.state_id==st)&(cost.G==G)&(cost.p==0.25)].dense_intersection_proxy.iloc[0])

states = sorted(cost.state_id.unique())
P, GAM = 0.25, 1.0
CFG = {"HWCS-1": (0.50, 0.50), "HWCS-2": (0.75, 1.0/3.0)}

def W_N1(rho):     return rho*(1 + GAM*P)
def W_N2(rho):     return P*rho*(2 + GAM)
def W_HWCS(rho,q,k): return 1.0 + (P + k*(q-P))*rho + GAM*P*rho

rows = []
for st in states:
    rho = dense_proxy(st,4)/dense_proxy(st,16)
    for name,(q,r) in CFG.items():
        for k in (0.0, 0.5, 1.0):
            w = W_HWCS(rho,q,k)
            rows.append(dict(
                state_id=st, method=name, kappa=k, rho=rho,
                W_N1=W_N1(rho), W_N2=W_N2(rho), W_HWCS=w,
                # vs N1
                hwcs_vs_N1_save=1 - w/W_N1(rho),
                beats_N1=bool(w < W_N1(rho)),
                # vs N2
                hwcs_vs_N2_save=1 - w/W_N2(rho),
                beats_N2=bool(w < W_N2(rho)),
                # N2 vs N1 (how much the strong baseline already saves)
                N2_vs_N1_save=1 - W_N2(rho)/W_N1(rho),
            ))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}\\stage_b_n1_n2_audit.csv", index=False)

pd.set_option("display.width", 240, "display.max_columns", 40,
              "display.float_format", lambda x: f"{x:.4f}")
print("="*82)
print("N1 vs N2 Native-G4 structural work audit (units of I_G16; PROXY not GPU time)")
print("="*82)
for k in (0.0, 0.5, 1.0):
    sub = df[df.kappa==k]
    print(f"\n--- kappa={k} ---")
    print(f"  HWCS beats N1: {int(sub.beats_N1.sum())}/{len(sub)} states "
          f"(mean save {sub.hwcs_vs_N1_save.mean():.3f})")
    print(f"  HWCS beats N2: {int(sub.beats_N2.sum())}/{len(sub)} states "
          f"(mean save {sub.hwcs_vs_N2_save.mean():.3f})")
print(f"\n  N2 vs N1 (strong baseline already saves, mean): {df[df.kappa==1].N2_vs_N1_save.mean():.3f}")

print("\n" + "="*82)
print("Per-state detail (kappa=1 conservative, kappa=0 optimistic)")
print("="*82)
show = df[df.kappa.isin([0.0,1.0])][["state_id","method","kappa","W_N1","W_N2","W_HWCS",
                                     "beats_N1","beats_N2","hwcs_vs_N2_save"]]
print(show.to_string(index=False))

# Summary verdict
k1 = df[df.kappa==1.0]; k0 = df[df.kappa==0.0]
verdict = {
  "evidence_class": "DERIVED_STRUCTURAL_PROXY_NOT_GPU_TIME",
  "N1_is_dense_bin_then_mask": True,
  "N2_is_mask_then_bin": True,
  "N2_vs_N1_mean_save": float(df[df.kappa==1].N2_vs_N1_save.mean()),
  "HWCS_vs_N1": {
     "kappa1_beats_states": int(k1.beats_N1.sum()), "of": int(len(k1)),
     "kappa1_mean_save": float(k1.hwcs_vs_N1_save.mean()),
     "kappa0_beats_states": int(k0.beats_N1.sum()), "of": int(len(k0)),
     "kappa0_mean_save": float(k0.hwcs_vs_N1_save.mean()),
  },
  "HWCS_vs_N2": {
     "kappa1_beats_states": int(k1.beats_N2.sum()), "of": int(len(k1)),
     "kappa1_mean_save": float(k1.hwcs_vs_N2_save.mean()),
     "kappa0_beats_states": int(k0.beats_N2.sum()), "of": int(len(k0)),
     "kappa0_mean_save": float(k0.hwcs_vs_N2_save.mean()),
  },
  "conclusion": ("HWCS's structural edge over N1 is modest and cost-model conditional. Against the "
                 "STRONG N2 baseline (mask-then-bin), HWCS does NOT reliably win: at kappa=1 it loses "
                 "in all states; only in the optimistic kappa=0 limit does it edge N2, and then only "
                 "marginally. Therefore no HWCS structural dominance can be claimed off N1 alone; "
                 "N2 must be beaten on MEASURED GPU time before any CODEX_GO discussion."),
}
json.dump(verdict, open(f"{OUT}\\stage_b_n1_n2_verdict.json","w"), indent=2)
print("\n" + json.dumps(verdict, indent=2))
