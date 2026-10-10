#!/usr/bin/env python3
"""
WV-HIER-R1-MX  Stage A — Part 2
Cost model + observed-NMSE calibration + HWCS prediction + equal-cost analysis.

INPUTS (OBSERVED / RECOMPUTED from the MX audit evidence package):
  WV0_ABSOLUTE_COST.csv  -> dense bbox intersection proxies I_G16, I_G4 (per state)
  WV0_PARETO_RECHECK.csv -> OBSERVED geometry-macro NMSE for G16/G8/G4/G1 at p=.25,.40

ALL quality numbers here are DERIVED from OBSERVED G16/G4 NMSE via the exact
variance-blend identity proven in stage_a_variance.py.  No raw HWCS gradient
exists; these are MODEL PREDICTIONS (prediction intervals), never measurements.
"""
import json
import numpy as np
import pandas as pd

EV = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx\evidence"
OUT = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx"

cost = pd.read_csv(f"{EV}\\WV0_ABSOLUTE_COST.csv")
par = pd.read_csv(f"{EV}\\WV0_PARETO_RECHECK.csv")

# --- dense intersection proxies per state (camera 0) -------------------------
def dense_proxy(state, G):
    row = cost[(cost.state_id == state) & (cost.G == G) & (cost.p == 0.25)]
    return float(row.dense_intersection_proxy.iloc[0])

states = sorted(cost.state_id.unique())

# --- observed NMSE lookup -----------------------------------------------------
def nmse(state, G, p):
    row = par[(par.state_id == state) & (par.G == G) & (np.isclose(par.p, p))]
    return float(row.nmse_geometry_macro.iloc[0])

P_Q = {"HWCS-1": (0.50, 0.50), "HWCS-2": (0.75, 1.0 / 3.0)}
P_MARG = 0.25  # both HWCS configs target marginal p = 0.25 (== G16/G4 baseline p)

rows = []
calib_rows = []
for st in states:
    I16 = dense_proxy(st, 16)
    I4 = dense_proxy(st, 4)
    rho = I4 / I16
    # observed NMSE at the common marginal p = 0.25
    n16 = nmse(st, 16, 0.25)
    n8 = nmse(st, 8, 0.25)
    n4 = nmse(st, 4, 0.25)
    n1 = nmse(st, 1, 0.25)
    # K calibration (Bernoulli NMSE = K*(1-p)/p) from BOTH p values, per G
    K = {}
    for G, nm in [(16, n16), (8, n8), (4, n4), (1, n1)]:
        n_hi = nmse(st, G, 0.40)
        K_lo = nm * 0.25 / 0.75          # K estimated from p=.25
        K_hi = n_hi * 0.40 / 0.60        # K estimated from p=.40
        K[G] = dict(K_from_p25=K_lo, K_from_p40=K_hi,
                     rel_inconsistency=abs(K_lo - K_hi) / abs(K_lo))
        K.setdefault("_val", {})[G] = K_lo
    # HWCS prediction via exact blend: NMSE_HWCS = (1-a)*NMSE_G4 + a*NMSE_G16
    hwcs_pred = {}
    for name, (q, r) in P_Q.items():
        p = q * r
        a = r * (1 - q) / (1 - p)
        hwcs_pred[name] = dict(
            q=q, r=r, p=p, alpha=a,
            nmse=(1 - a) * n4 + a * n16,
            # 90% prediction band: blend the two endpoint NMSEs' own spread is
            # unavailable (single observed mean each), so band uses the
            # G16-vs-G4 gap as the structural uncertainty scale.
            nmse_lo=(1 - a) * n4 + a * n16,  # point; band added below
        )
    # structural uncertainty band = +/- (alpha-weighted half-gap to G16)
    gap = n16 - n4
    for name in hwcs_pred:
        a = hwcs_pred[name]["alpha"]
        hwcs_pred[name]["nmse_band"] = a * gap * 0.5  # +/- half of alpha*gap
    rows.append(dict(state_id=st, I_G16=I16, I_G4=I4, rho=I4 / I16,
                     nmse_G16=n16, nmse_G8=n8, nmse_G4=n4, nmse_G1=n1,
                     hwcs1_nmse=hwcs_pred["HWCS-1"]["nmse"],
                     hwcs2_nmse=hwcs_pred["HWCS-2"]["nmse"],
                     hwcs1_alpha=hwcs_pred["HWCS-1"]["alpha"],
                     hwcs2_alpha=hwcs_pred["HWCS-2"]["alpha"],
                     hwcs1_band=hwcs_pred["HWCS-1"]["nmse_band"],
                     hwcs2_band=hwcs_pred["HWCS-2"]["nmse_band"]))
    for G in (16, 8, 4, 1):
        calib_rows.append(dict(state_id=st, G=G, **K[G]))

pred = pd.DataFrame(rows)
calib = pd.DataFrame(calib_rows)

# ---------------------------------------------------------------------------
# WORK MODEL  (structural proxy; NOT measured CUDA time)
#   binning+sort entries (dominant tile-rasterizer cost), units of I_G16:
#     G16      : 1                 (dense parent bin+sort)  [+ raster p, small]
#     nativeG4 : rho               (dense child bin+sort)
#     HWCS cons: 1 + q*rho         (parent bin+sort + child refine in active parents)
#     HWCS opt : 1 + p*rho         (parent bin+sort + only selected children via filter)
#   raster (selected-children) is identical for nativeG4 and HWCS at same p.
# ---------------------------------------------------------------------------
GAM = 1.0  # raster cost per selected-child entry relative to one bin entry
def work(method, rho, p=0.25, q=0.5):
    if method == "G16":
        return 1.0 + GAM * p            # parent bin (1) + raster p (units I_G16)
    if method == "G4":
        return rho * (1.0 + GAM * p)    # dense child bin (rho) + raster p*rho
    if method == "HWCS_cons":
        return 1.0 + q * rho + GAM * p * rho
    if method == "HWCS_opt":
        return 1.0 + p * rho + GAM * p * rho
    raise ValueError

work_rows = []
for _, r in pred.iterrows():
    rho = r.rho
    d = dict(state_id=r.state_id, rho=rho,
             W_G16=work("G16", rho), W_G4=work("G4", rho),
             W_HWCS1_cons=work("HWCS_cons", rho, q=0.50),
             W_HWCS2_cons=work("HWCS_cons", rho, q=0.75),
             W_HWCS1_opt=work("HWCS_opt", rho, q=0.50),
             W_HWCS2_opt=work("HWCS_opt", rho, q=0.75))
    d["HWCS1_save_vs_G4_cons"] = 1 - d["W_HWCS1_cons"] / d["W_G4"]
    d["HWCS2_save_vs_G4_cons"] = 1 - d["W_HWCS2_cons"] / d["W_G4"]
    d["HWCS1_save_vs_G4_opt"] = 1 - d["W_HWCS1_opt"] / d["W_G4"]
    work_rows.append(d)
work_df = pd.DataFrame(work_rows)

# ---------------------------------------------------------------------------
# EQUAL-COST DOMINANCE
#   Question: at a fixed absolute structural work W, is any HWCS point below
#   the straight segment joining G16 (low work, high NMSE) and native G4
#   (high work, low NMSE) in (work, NMSE) space?  If HWCS sits ABOVE the
#   segment it is dominated by a G16/G4 mixture; BELOW => genuine advantage.
#   We use the CONSERVATIVE work model (child refine in active parents).
# ---------------------------------------------------------------------------
dom_rows = []
for _, r in pred.iterrows():
    rho = r.rho
    pts = {}
    pts["G16"] = (work("G16", rho), r.nmse_G16)
    pts["G4"] = (work("G4", rho), r.nmse_G4)
    (w16, v16), (w4, v4) = pts["G16"], pts["G4"]
    def seg_nmse(w):  # NMSE on the G16-G4 straight segment at work w
        if w4 == w16:
            return v16
        t = (w - w16) / (w4 - w16)
        t = min(max(t, 0.0), 1.0)
        return v16 + t * (v4 - v16)
    for name, wk, vk in [("HWCS-1", r.W_HWCS1_cons if hasattr(r, "W_HWCS1_cons") else work("HWCS_cons", rho, 0.5), r.hwcs1_nmse),
                         ("HWCS-2", work("HWCS_cons", rho, 0.75), r.hwcs2_nmse)]:
        seg = seg_nmse(wk)
        dom_rows.append(dict(state_id=r.state_id, method=name, work=wk,
                             nmse=vk, segment_nmse_at_same_work=seg,
                             delta=seg - vk,   # >0 => HWCS BELOW segment (good)
                             verdict=("HWCS_BELOW_SEGMENT" if vk < seg - 1e-9
                                      else "HWCS_ABOVE_SEGMENT_DOMINATED")))
dom = pd.DataFrame(dom_rows)

# ---------------------------------------------------------------------------
# EQUAL-COST NMSE ratio (reproduce audit's G4/G16 at matched proxy cost, then
# extend to HWCS).  At cost C = p*I_G16 (G16 p=.25 budget):
#   each method's affordable p = C / I_G ; NMSE = K_G*(1/p - 1)
# ---------------------------------------------------------------------------
eq_rows = []
for _, r in pred.iterrows():
    rho = r.rho
    K16 = r.nmse_G16 * 0.25 / 0.75
    K4 = r.nmse_G4 * 0.25 / 0.75
    C = 0.25 * r.I_G16  # G16 p=.25 absolute proxy budget
    # affordable p per method at this cost
    p16 = C / r.I_G16
    p4 = C / r.I_G4
    # HWCS: keep marginal p (blend) -> report its NMSE at its own p=.25 and the
    # work it costs, so the comparison is cost-matched via the work model instead.
    n16_eq = K16 * (1 / p16 - 1)
    n4_eq = K4 * (1 / p4 - 1)
    eq_rows.append(dict(state_id=r.state_id, rho=rho,
                        G4_over_G16_eqcost=n4_eq / n16_eq,
                        note="audit reproduced: >1 means G16 cheaper at matched cost"))
eq = pd.DataFrame(eq_rows)

# ---------------------------------------------------------------------------
pred.to_csv(f"{OUT}\\stage_a_hwcs_predictions.csv", index=False)
work_df.to_csv(f"{OUT}\\stage_a_cost_model.csv", index=False)
dom.to_csv(f"{OUT}\\stage_a_dominance.csv", index=False)
calib.to_csv(f"{OUT}\\stage_a_K_calibration.csv", index=False)
eq.to_csv(f"{OUT}\\stage_a_equal_cost_ratio.csv", index=False)

# ---- console summary ----
pd.set_option("display.width", 200, "display.max_columns", 40,
              "display.float_format", lambda x: f"{x:.4f}")
print("=" * 78)
print("STAGE A PART 2 — K calibration consistency (Bernoulli model fit)")
print("=" * 78)
print(calib.groupby("G").rel_inconsistency.agg(["mean", "max"]).rename(
    columns={"mean": "mean_rel_inconsistency", "max": "max_rel_inconsistency"}))

print("\n" + "=" * 78)
print("HWCS NMSE prediction (marginal p=0.25) vs G16 / native G4")
print("=" * 78)
show = pred[["state_id", "rho", "nmse_G16", "hwcs2_nmse", "hwcs1_nmse", "nmse_G4"]].copy()
print(show.to_string(index=False))

print("\n" + "=" * 78)
print("Structural WORK (conservative child-refine model), units of I_G16")
print("=" * 78)
print(work_df[["state_id", "rho", "W_G16", "W_HWCS1_cons", "W_HWCS2_cons",
               "W_G4", "HWCS1_save_vs_G4_cons", "HWCS2_save_vs_G4_cons"]].to_string(index=False))

print("\n" + "=" * 78)
print("EQUAL-COST DOMINANCE (conservative work; delta>0 => HWCS below G16-G4 segment)")
print("=" * 78)
print(dom.to_string(index=False))
print("\nVerdict tally:")
print(dom.groupby(["method", "verdict"]).size())

print("\n" + "=" * 78)
print("Audit reproduction: G4/G16 NMSE at matched absolute proxy cost")
print("=" * 78)
print(eq[["state_id", "rho", "G4_over_G16_eqcost"]].to_string(index=False))
print(f"\nrange G4/G16 eqcost: {eq.G4_over_G16_eqcost.min():.4f} .. {eq.G4_over_G16_eqcost.max():.4f}")
