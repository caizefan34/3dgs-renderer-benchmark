#!/usr/bin/env python3
"""
WV-HIER-R1-MX — Variance-identity MC check (formal deliverable artifact).
Produces variance_identity_check.csv: per-config empirical vs theoretical variance
of the HT estimator, with relative error, plus the per-pixel inclusion frequency
and the signed within-parent cross-term s2-s1 (to document the scene-dependence
of the G4<=HWCS<=G16 ordering).

This is a CPU/numpy Monte Carlo of the ESTIMATOR MATH (not a GPU gradient run).
It validates the identity  V_HWCS = (1-alpha) V_G4 + alpha V_G16  that Stage B-min
must confirm on real gradients. A finite-sample deviation of a single seed is NOT
a formula failure; only the paired CI on the mean difference matters.
"""
import csv, math, os
import numpy as np

OUT = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx"
P, GAM = 0.25, 1.0
CFG = {"G16": (P,1.0), "G4": (1.0,P), "HWCS-1": (0.5,0.5), "HWCS-2": (0.75,1.0/3.0)}

def run(seed_contrib, n, seed):
    """seed_contrib: (parents, children) array of per-child gradient contributions
    (may be signed). Returns per-config empirical variance + inclusion freq."""
    rng = np.random.default_rng(seed)
    Pp, C = seed_contrib.shape
    s1 = float((seed_contrib**2).sum())
    b = seed_contrib.sum(1); s2 = float((b**2).sum())
    vg4 = (1-P)/P*s1; vg16 = (1-P)/P*s2
    true = float(seed_contrib.sum())
    up = rng.random((n, Pp)); uc = rng.random((n, Pp, C))
    out = []
    for m,(q,r) in CFG.items():
        par = (up < q)[:,:,None]; ch = (uc < r)
        xi = par & ch
        est = (xi*seed_contrib).sum((1,2))/P
        empv = float(((est-true)**2).mean())
        empmean = float(est.mean())
        alpha = r*(1-q)/(1-P)
        thv = (1-alpha)*vg4 + alpha*vg16
        out.append(dict(method=m, q=q, r=r, alpha=alpha,
            empirical_variance=empv, theoretical_variance=thv,
            rel_err=abs(empv-thv)/abs(thv),
            mean_abs_err=abs(empmean-true),
            s1=s1, s2=s2, s2_minus_s1=s2-s1))
    return out

# Three contribution regimes: positive (s2>s1), mixed (s2~s1), negative (s2<s1)
regimes = {
  "positive_same_sign": lambda Pc,C: (np.arange(Pc)[:,None]*1.0 + np.arange(C)[None,:]*0.1),
  "mixed_signed":       lambda Pc,C: (((np.arange(Pc)[:,None]*17 + np.arange(C)[None,:]*7) % 23) - 11)/13.0,
  "negative_antialign": lambda Pc,C: (np.sin(np.arange(Pc)[:,None]*1.7)*np.cos(np.arange(C)[None,:]*0.9)*3.0),
}
n = 300000
rows = []
for rname, gen in regimes.items():
    for (Pc,C) in [(6,16),(12,16),(40,16)]:
        a = gen(Pc,C)
        for rec in run(a, n, seed=abs(hash(rname))%100000 + Pc*100 + C):
            rec["regime"]=rname; rec["parents"]=Pc; rec["children_per_parent"]=C
            rows.append(rec)

cols = ["regime","parents","children_per_parent","method","q","r","alpha",
        "empirical_variance","theoretical_variance","rel_err","mean_abs_err",
        "s1","s2","s2_minus_s1"]
with open(f"{OUT}\\variance_identity_check.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow(r)

# aggregate
import pandas as pd
df = pd.DataFrame(rows)
print("Variance-identity MC check (n=%d draws/config)" % n)
print("="*70)
print("Relative error |emp-theory|/theory by method (max over regimes):")
print(df.groupby("method").rel_err.agg(["mean","max"]).rename(columns={"mean":"mean_relerr","max":"max_relerr"}))
print("\nAll rel_err < 0.05 ?", bool((df.rel_err < 0.05).all()))
print("\ns2-s1 sign by regime (documents ordering scene-dependence):")
print(df.groupby("regime").s2_minus_s1.first())
print("\nWrote variance_identity_check.csv (%d rows)" % len(rows))
