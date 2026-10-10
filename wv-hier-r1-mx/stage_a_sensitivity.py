#!/usr/bin/env python3
"""
WV-HIER-R1-MX  Stage A — Part 3
Sensitivity of the HWCS verdict to the (UNMEASURED) child-refinement cost,
plus the directly-relevant matched-work / matched-NMSE comparison vs native G4.

Child-refine efficiency kappa in [0,1]:
    child_refine_work(kappa) = p*rho + kappa*(q - p)*rho
      kappa = 1 -> conservative (refine ALL children in active parents) = q*rho
      kappa = 0 -> optimistic  (touch ONLY selected children, free filter) = p*rho
Total structural work (units of I_G16, GAM=1):
    W(method) = bin + raster
      G16       : 1 + GAM*p
      G4        : rho + GAM*p*rho                (native G4 dense child binning)
      HWCS(k)   : 1 + [p*rho + kappa*(q-p)*rho] + GAM*p*rho
"""
import numpy as np
import pandas as pd

EV = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx\evidence"
OUT = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx"
cost = pd.read_csv(f"{EV}\\WV0_ABSOLUTE_COST.csv")
par = pd.read_csv(f"{EV}\\WV0_PARETO_RECHECK.csv")

states = sorted(cost.state_id.unique())
def dp(st, G, p=0.25):
    return float(cost[(cost.state_id == st) & (cost.G == G) & (cost.p == p)].dense_intersection_proxy.iloc[0])
def nm(st, G, p):
    return float(par[(par.state_id == st) & (par.G == G) & (np.isclose(par.p, p))].nmse_geometry_macro.iloc[0])

GAM = 1.0
PG = 0.25
CFG = {"HWCS-1": (0.50, 0.50), "HWCS-2": (0.75, 1.0 / 3.0)}

def W_G4(rho, p):     return rho + GAM * p * rho
def W_G16(rho, p):    return 1.0 + GAM * p
def W_HWCS(rho, p, q, kappa):
    return 1.0 + (p * rho + kappa * (q - p) * rho) + GAM * p * rho
def nmse_G4_at_p(rho, K4, p):   # K4*(1/p - 1)
    return K4 * (1.0 / p - 1.0)
def p_G4_at_work(rho, W):
    # W = rho + GAM*p*rho  -> p = (W/rho - 1)/GAM
    return (W / rho - 1.0) / GAM

rows = []
for st in states:
    rho = dp(st, 4) / dp(st, 16)
    K4 = nm(st, 4, 0.25) * 0.25 / 0.75
    n16, n4 = nm(st, 16, 0.25), nm(st, 4, 0.25)
    g4_floor = W_G4(rho, 0.0)          # native G4 min work (dense bin, p=0)
    for name, (q, r) in CFG.items():
        a = r * (1 - q) / (1 - q * r)
        n_hwcs = (1 - a) * n4 + a * n16
        for kappa, ktag in [(1.0, "cons"), (0.0, "opt"), (0.5, "mid")]:
            W = W_HWCS(rho, PG, q, kappa)
            # matched-work: G4 NMSE at this work (if reachable)
            p4w = p_G4_at_work(rho, W)
            if 0 < p4w <= 1:
                n4_mw = nmse_G4_at_p(rho, K4, p4w)
                mw_verdict = "HWCS_lower_NMSE" if n_hwcs < n4_mw else "G4_lower_NMSE"
                mw_delta = n4_mw - n_hwcs
            else:
                n4_mw = float("nan"); mw_verdict = "G4_work_floor_above_HWCS"; mw_delta = float("nan")
            # work saving vs G4 at p=.25 (reference)
            save = 1 - W / W_G4(rho, PG)
            rows.append(dict(state_id=st, method=name, kappa=ktag, rho=rho,
                             W_HWCS=W, W_G4_floor=g4_floor, W_G4_p25=W_G4(rho, PG),
                             nmse_HWCS=n_hwcs, nmse_G4_p25=n4,
                             p4_for_matched_work=p4w, nmse_G4_matched_work=n4_mw,
                             matched_work_verdict=mw_verdict, matched_work_delta=mw_delta,
                             work_save_vs_G4_p25=save))
sens = pd.DataFrame(rows)

# ---- break-even kappa where HWCS-1 work == native G4 work floor (rho) --------
# W_HWCS(kappa) = g4_floor  ->  solve kappa
be = []
for st in states:
    rho = dp(st, 4) / dp(st, 16)
    for name, (q, r) in CFG.items():
        # 1 + p*rho + kappa*(q-p)*rho + p*rho = rho
        base = 1.0 + PG * rho + GAM * PG * rho
        coef = (q - PG) * rho
        if abs(coef) < 1e-12:
            kstar = float("nan")
        else:
            kstar = (rho - base) / coef
        be.append(dict(state_id=st, method=name, rho=rho, kappa_star_G4floor=kstar,
                       note="kappa<kstar => HWCS below G4 work floor"))
breakeven = pd.DataFrame(be)

# ---- segment (mixture) dominance as function of kappa ------------------------
seg_rows = []
for st in states:
    rho = dp(st, 4) / dp(st, 16)
    n16, n4 = nm(st, 16, 0.25), nm(st, 4, 0.25)
    w16, w4 = W_G16(rho, PG), W_G4(rho, PG)
    def seg(w):
        t = (w - w16) / (w4 - w16); t = min(max(t, 0), 1)
        return n16 + t * (n4 - n16)
    for name, (q, r) in CFG.items():
        a = r * (1 - q) / (1 - q * r)
        n_hwcs = (1 - a) * n4 + a * n16
        for kappa in [1.0, 0.75, 0.5, 0.25, 0.0]:
            W = W_HWCS(rho, PG, q, kappa)
            s = seg(W)
            seg_rows.append(dict(state_id=st, method=name, kappa=kappa, W=W,
                                 nmse_HWCS=n_hwcs, segment_nmse=s,
                                 delta=s - n_hwcs,
                                 verdict=("BELOW_mixture" if n_hwcs < s - 1e-9
                                          else "ABOVE_mixture_dominated")))
seg = pd.DataFrame(seg_rows)

sens.to_csv(f"{OUT}\\stage_a_sensitivity.csv", index=False)
breakeven.to_csv(f"{OUT}\\stage_a_breakeven.csv", index=False)
seg.to_csv(f"{OUT}\\stage_a_segment_sweep.csv", index=False)

pd.set_option("display.width", 220, "display.max_columns", 40,
              "display.float_format", lambda x: f"{x:.4f}")
print("=" * 80)
print("MATCHED-WORK: HWCS vs native G4 at equal structural work (kappa=1 conservative)")
print("=" * 80)
c = sens[(sens.kappa == "cons")]
print(c[["state_id", "method", "W_HWCS", "W_G4_floor", "nmse_HWCS",
         "nmse_G4_matched_work", "matched_work_verdict", "matched_work_delta",
         "work_save_vs_G4_p25"]].to_string(index=False))
print("\nMatched-work verdict tally (conservative):")
print(c.groupby(["method", "matched_work_verdict"]).size())

print("\n" + "=" * 80)
print("WORK SAVING vs native G4 (p=.25 reference)")
print("=" * 80)
print(sens[sens.kappa == "cons"].groupby("method").work_save_vs_G4_p25.agg(
    ["mean", "min", "max"]))

print("\n" + "=" * 80)
print("BREAK-EVEN kappa where HWCS work == native G4 dense-binning work floor")
print("=" * 80)
print(breakeven.groupby("method").kappa_star_G4floor.agg(["mean", "min", "max"]))
print("(kappa < kstar => HWCS strictly below G4's minimum work; all states)")

print("\n" + "=" * 80)
print("SEGMENT (G16/G4 mixture) dominance vs kappa  [BELOW_mixture => genuine]")
print("=" * 80)
piv = seg.pivot_table(index=["method", "kappa"], columns="verdict",
                      values="state_id", aggfunc="count").fillna(0)
print(piv)
print("\nPer-kappa net (states where BELOW minus states where ABOVE):")
net = seg.assign(bit=np.where(seg.verdict == "BELOW_mixture", 1, -1)) \
         .groupby(["method", "kappa"]).bit.agg(["sum", "count"])
print(net)
