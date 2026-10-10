#!/usr/bin/env python3
"""
WV-HIER-R1-MX — Round 4: cost-model sensitivity to N2 overhead and parent-binning cost.

Rounds 1-3 established (GAM-invariant): at kappa=1, HWCS loses to N2 in all states.
That proof assumed N2 (mask-then-bin) has NO implementation overhead (o_N2=1) and
parent binning costs exactly 1 unit (c_par=1). A real CUDA N2 may incur overhead
(two-pass, load imbalance, extra launch); a real HWCS parent pass may cost >1 unit.

This sweep varies:
  c_par  in [0,3]   : parent binning cost, in I_G16 units (HWCS and G16 pay this)
  o_N2   in [1,2]   : N2 overhead multiplier (1 = ideal mask-then-bin)
and asks: for which (c_par, o_N2) does HWCS beat N2 at kappa=1? Does it beat N1?

W_N1   = c_par*rho*(1 + GAM*p)/1        [N1 pays dense child binning at c_par per G16-unit... see note]
   -- to keep units clean we model all binning at the SAME per-entry cost c_par:
W_N1   = c_par * rho * (1 + GAM*p)
W_N2   = o_N2 * c_par * p*rho*(2 + GAM)
W_HWCS = c_par*1 + [p + kappa*(q-p)]*rho + GAM*p*rho     (HWCS pays c_par for the 1 parent pass)

At kappa=1, HWCS-1 (q-p=0.25):  W_HWCS = c_par + 0.5*rho + GAM*p*rho
"""
import json, os
import numpy as np
import pandas as pd

EV = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx\evidence"
OUT = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx"
cost = pd.read_csv(f"{EV}\\WV0_ABSOLUTE_COST.csv")
states = sorted(cost.state_id.unique())
def dp(st,G): return float(cost[(cost.state_id==st)&(cost.G==G)&(cost.p==0.25)].dense_intersection_proxy.iloc[0])
P, GAM = 0.25, 1.0
CFG = {"HWCS-1": (0.5,0.5), "HWCS-2": (0.75,1.0/3.0)}

C_PAR = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
O_N2  = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0]

def W_N1(rho, c): return c*rho*(1+GAM*P)
def W_N2(rho, c, o): return o*c*P*rho*(2+GAM)
def W_HWCS(rho, c, q, k): return c*1.0 + (P + k*(q-P))*rho + GAM*P*rho

rows = []
for st in states:
    rho = dp(st,4)/dp(st,16)
    for name,(q,r) in CFG.items():
        for c in C_PAR:
            for o in O_N2:
                w = W_HWCS(rho,c,q,1.0)
                rows.append(dict(state_id=st, method=name, c_par=c, o_N2=o, rho=rho,
                    W_HWCS=w, W_N1=W_N1(rho,c), W_N2=W_N2(rho,c,o),
                    beats_N1=bool(w < W_N1(rho,c)),
                    beats_N2=bool(w < W_N2(rho,c,o))))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}\\stage_b_cost_model_sensitivity.csv", index=False)

# For each (c_par, o_N2): fraction of states where HWCS beats N2 (kappa=1)
piv = df.groupby(["c_par","o_N2"]).agg(
    frac_beat_N2=("beats_N2","mean"), frac_beat_N1=("beats_N1","mean")).reset_index()

print("Cost-model sensitivity: fraction of 9 states where HWCS beats N2 at kappa=1")
print("(rows = parent-binning cost c_par; cols = N2 overhead multiplier o_N2)")
print("="*84)
for name in CFG:
    sub = df[df.method==name]
    p = sub.groupby(["c_par","o_N2"]).beats_N2.mean().unstack()
    p = p.reindex(index=C_PAR, columns=O_N2)
    print(f"\n--- {name}: P(beat N2) over 9 states ---")
    print(p.map(lambda x: f"{x:.2f}").to_string())

# critical o_N2 for HWCS-1 to beat N2 at c_par=1, kappa=1, per state
print("\n--- critical o_N2 (min N2 overhead for HWCS-1 to beat N2, c_par=1, kappa=1) ---")
crit = []
for st in states:
    rho = dp(st,4)/dp(st,16)
    # HWCS < N2  <=>  c_par + 0.5*rho + GAM*p*rho < o*c_par*p*rho*(2+GAM)
    # solve for o:  o > (c_par + 0.5*rho + GAM*p*rho) / (c_par*p*rho*(2+GAM))
    c=1.0
    num = c + 0.5*rho + GAM*P*rho
    den = c*P*rho*(2+GAM)
    o_crit = num/den
    crit.append(dict(state_id=st, rho=rho, o_N2_critical=float(o_crit)))
cdf = pd.DataFrame(crit)
print(cdf.to_string(index=False))
print(f"\n  o_N2_critical range: {cdf.o_N2_critical.min():.3f} .. {cdf.o_N2_critical.max():.3f}")
print(f"  => HWCS-1 beats N2 at kappa=1 ONLY IF N2 overhead multiplier > this value.")
print(f"     At o_N2=1 (ideal), it never beats N2 (matches round-2 GAM-invariance).")

verdict = {
  "evidence_class": "DERIVED_COST_MODEL_SENSITIVITY",
  "swept": {"c_par": C_PAR, "o_N2": O_N2, "kappa": 1.0},
  "key_unknown": "N2 (mask-then-bin) implementation overhead o_N2 and parent-binning cost c_par",
  "finding_at_ideal": "At o_N2=1, c_par=1: HWCS loses to N2 in all states (round-2 result, GAM-invariant)",
  "finding_with_overhead": "HWCS beats N2 at kappa=1 ONLY IF o_N2 exceeds a per-state critical threshold",
  "o_N2_critical_range_H1": [float(cdf.o_N2_critical.min()), float(cdf.o_N2_critical.max())],
  "conclusion": ("The 'HWCS loses to N2' result is robust to GAM and c_par, but NOT to N2 overhead. "
     "If real N2 mask-then-bin incurs >{lo:.2f}-{hi:.2f}x overhead (two-pass, load imbalance, extra launch), "
     "HWCS-1 can beat N2 at kappa=1. This makes N2's real CUDA overhead the KEY unmeasured quantity that "
     "Stage B-min structural counts and Stage C timing must pin down. It does NOT restore a CODEX_GO: the "
     "overhead is itself unmeasured, and a count-based proxy cannot establish it.").format(
        lo=cdf.o_N2_critical.min(), hi=cdf.o_N2_critical.max()),
}
json.dump(verdict, open(f"{OUT}\\stage_b_cost_model_sensitivity.json","w"), indent=2)
print("\n" + json.dumps(verdict, indent=2))
