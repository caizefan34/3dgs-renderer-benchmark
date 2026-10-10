#!/usr/bin/env python3
"""
WV-HIER-R1-MX — Round 2: GAM-robustness of the N1/N2 dominance conclusion.

Round-1 N1/N2 audit used GAM=1 (raster weight). The goal requires the Native-G4
floor advantage to be limited to "N1 + the current structural cost model". A
natural challenge: is the "HWCS loses to N2 at kappa=1" conclusion an artifact of
GAM=1? This script proves (analytically AND numerically) that it is NOT: GAM
cancels out of the HWCS-vs-N1 and HWCS-vs-N2 comparisons, giving analytic
break-even kappa thresholds that depend only on rho.

Derivation (work in units of I_G16, p=0.25):
  W_N1   = rho*(1 + GAM*p)
  W_N2   = p*rho*(2 + GAM)
  W_HWCS = 1 + [p + kappa*(q-p)]*rho + GAM*p*rho
HWCS < N1  <=>  1 < rho*[1 - p - kappa*(q-p)]        (GAM cancels)
HWCS < N2  <=>  1 < rho*[p - kappa*(q-p)]            (GAM cancels)
Break-even kappa:
  kappa*_N1 = (1 - p - 1/rho)/(q-p)     [HWCS beats N1 for kappa < kappa*_N1]
  kappa*_N2 = (p - 1/rho)/(q-p)         [HWCS beats N2 for kappa < kappa*_N2]
"""
import json, os
import numpy as np
import pandas as pd

EV = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx\evidence"
OUT = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx"
cost = pd.read_csv(f"{EV}\\WV0_ABSOLUTE_COST.csv")
states = sorted(cost.state_id.unique())
def dp(st, G): return float(cost[(cost.state_id==st)&(cost.G==G)&(cost.p==0.25)].dense_intersection_proxy.iloc[0])
P = 0.25
CFG = {"HWCS-1": (0.50, 0.50), "HWCS-2": (0.75, 1.0/3.0)}

# analytic break-even kappa vs N2 (GAM-independent)
rows = []
for st in states:
    rho = dp(st,4)/dp(st,16)
    for name,(q,r) in CFG.items():
        qmp = q - P
        kstar_n2 = (P - 1.0/rho)/qmp if qmp > 0 else float('nan')
        kstar_n1 = (1 - P - 1.0/rho)/qmp if qmp > 0 else float('nan')
        rows.append(dict(state_id=st, method=name, rho=rho,
            kappa_star_N2=kstar_n2, kappa_star_N1=kstar_n1,
            beats_N2_at_k1=bool(kstar_n2 > 1.0),   # at kappa=1, HWCS<N2 iff 1<kstar_N2
            beats_N1_at_k1=bool(kstar_n1 > 1.0)))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}\\stage_b_n1_n2_gam_robustness.csv", index=False)

pd.set_option("display.width", 220, "display.float_format", lambda x: f"{x:.4f}")
print("="*78)
print("GAM-robustness: break-even kappa (GAM cancels; depends only on rho)")
print("="*78)
print(df[["state_id","method","rho","kappa_star_N2","kappa_star_N1","beats_N2_at_k1","beats_N1_at_k1"]].to_string(index=False))

print("\n--- Analytic vs numeric cross-check (GAM=1 and GAM=3 must agree) ---")
def W_N1(rho,g): return rho*(1+g*P)
def W_N2(rho,g): return P*rho*(2+g)
def W_HWCS(rho,q,k,g): return 1 + (P + k*(q-P))*rho + g*P*rho
for st in ["bicycle__iter15000","room__iter5000"]:
    rho = dp(st,4)/dp(st,16)
    for name,(q,r) in CFG.items():
        for g in (1.0, 3.0):
            w1 = W_HWCS(rho,q,1.0,g); wn2 = W_N2(rho,g); wn1 = W_N1(rho,g)
            print(f"  {st[:18]:18s} {name} GAM={g}: HWCS<N2={w1<wn2} HWCS<N1={w1<wn1}")

# numeric break-even confirmation: solve W_HWCS(kappa)=W_N2 for kappa, both GAM
print("\n--- Numeric break-even kappa (root find) vs analytic, GAM=1 and GAM=3 ---")
def W_N1(rho,g): return rho*(1+g*P)
def W_N2(rho,g): return P*rho*(2+g)
def W_HWCS(rho,q,k,g): return 1 + (P + k*(q-P))*rho + g*P*rho
def bisect_kappa(rho,q,g,target,lo=0.0,hi=5.0,it=200):
    f=lambda k: W_HWCS(rho,q,k,g)-target
    for _ in range(it):
        mid=(lo+hi)/2
        if f(lo)*f(mid)<=0: hi=mid
        else: lo=mid
    return (lo+hi)/2
for st in ["bicycle__iter15000","room__iter5000"]:
    rho = dp(st,4)/dp(st,16)
    for name,(q,r) in CFG.items():
        qmp=q-P
        an = (P - 1.0/rho)/qmp
        for g in (1.0,3.0):
            num = bisect_kappa(rho,q,g,W_N2(rho,g))
            print(f"  {st[:18]:18s} {name} GAM={g}: kappa*_N2 numeric={num:.4f} analytic={an:.4f} "
                  f"match={abs(num-an)<1e-3}")

verdict = {
  "evidence_class": "DERIVED_ANALYTIC_PLUS_NUMERIC",
  "claim": "The HWCS-vs-N1 and HWCS-vs-N2 dominance conclusions are INDEPENDENT of the raster weight GAM",
  "proof": "GAM*p*rho appears on both sides of each comparison and cancels; break-even kappa depends only on rho",
  "kappa_star_N2_range": [float(df.kappa_star_N2.min()), float(df.kappa_star_N2.max())],
  "kappa_star_N1_range": [float(df.kappa_star_N1.min()), float(df.kappa_star_N1.max())],
  "beats_N2_at_kappa1_states": int(df.beats_N2_at_k1.sum()),
  "of": int(len(df)),
  "beats_N1_at_kappa1_states": int(df.beats_N1_at_k1.sum()),
  "conclusion": ("HWCS beats N2 only for kappa < (p - 1/rho)/(q-p) = "
                 f"{df.kappa_star_N2.min():.3f}..{df.kappa_star_N2.max():.3f} across states; "
                 "at kappa=1 it loses to N2 in ALL states, independent of GAM. HWCS beats N1 for "
                 f"kappa < (1-p-1/rho)/(q-p) = {df.kappa_star_N1.min():.3f}..{df.kappa_star_N1.max():.3f} "
                 "(all >1 at kappa=1). The N2-conditional weakness is therefore NOT a raster-weight artifact."),
}
json.dump(verdict, open(f"{OUT}\\stage_b_n1_n2_gam_robustness.json","w"), indent=2)
print("\n" + json.dumps(verdict, indent=2))
