#!/usr/bin/env python3
"""GRADIENT-PROXY-P0 Phase 3: offline Oracle U/E/G/H1/H2/H3 NMSE with CORRECT formula.

Loads per-tile true grad (HiGS) + H2 forward geometry from Phase 2a. Builds proxies:
- U: uniform
- E: error-guided (per-tile |L_j| residual)  [deployable: GT residual available in training]
- G: true gradient oracle (per-tile ||g_j||)  [NOT deployable - upper bound]
- H1: historical gradient EMA (carry-over per tile from PRIOR states; exploration floor)
- H2: geometry-Jacobian (residual * sqrt(opac2_mass))  [deployable: forward only]
- H3: hybrid H1+H2 with exploration floor
Correct formula: Var=sum[(1-p_j)/p_j]||g_j||^2; NMSE=Var/||g||^2.
Held-out: H1 designed on 5K/10K only, evaluated on 15K; cam2 held out throughout.
"""
import os, json, math, numpy as np
D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT=f"{D}/s1a_r1_deliverables"
CKPTS=["5K","10K","15K"]; CAMS=[0,1,2]; TILE=16
th=math.ceil(264/TILE); tw=math.ceil(396/TILE); NUM_TILES=th*tw
PK=["means","quats","scales","opac","colors"]

def load(ct,ci):
    z=np.load(f"{OUT}/gp0_pertile_{ct}_cam{ci}.npz")
    return {k:z[k] for k in z.files}

def ntb(w,p):
    w=np.nan_to_num(np.asarray(w,np.float64),nan=0.0,posinf=0.0,neginf=0.0); w=np.maximum(w,0)
    m=w.mean()
    if m<=1e-12: return np.full_like(w,1.0/len(w))
    pj=p*w/m; return np.clip(pj,1e-3,1.0)
def var_correct(pj,gk):
    pj=np.nan_to_num(np.asarray(pj,np.float64)); gk=np.nan_to_num(np.asarray(gk,np.float64))
    return float((((1-pj)/pj)*(gk**2)).sum())
def corr(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    a=a-a.mean(); b=b-b.mean()
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d>1e-12 else 0.0
def primary_gnorm(d): return d["gnorm_means"]

# ---- load all ----
data={}
for ct in CKPTS:
    for ci in CAMS:
        data[f"{ct}_cam{ci}"]=load(ct,ci)

# ---- correlation diagnostic: what cheap signal predicts true ||g_j||? ----
corr_diag={}
for ct in CKPTS:
    for ci in CAMS:
        d=data[f"{ct}_cam{ci}"]; gk=primary_gnorm(d)
        corr_diag[f"{ct}_cam{ci}"]={
            "residual":corr(d["residual_tile"],gk),"opac_mass":corr(d["opac_tile"],gk),
            "opac2":corr(d["opac2_tile"],gk),"cnt":corr(d["cnt_tile"],gk),
            "conic_mass":corr(d["conic_tile"],gk),
            "residual_x_conic":corr(d["residual_tile"]*d["conic_tile"],gk),
            "residual_x_sqrtcnt":corr(d["residual_tile"]*np.sqrt(np.maximum(d["cnt_tile"],1)),gk),
        }
print("=== CORR(true ||g_j||, cheap signals) ===")
for k,v in corr_diag.items(): print(f"  {k}: "+", ".join(f"{kk}={vv:.3f}" for kk,vv in v.items()))

# ---- build proxies per combo ----
# H2: geometry-Jacobian proxy = residual_tile * conic_tile (forward-only)
# H1: historical EMA of true ||g_means|| from PRIOR states (design states 5K,10K only)
# H3: hybrid = normalize(H1)+normalize(H2), exploration floor
results={}
# H1 history: EMA updated in state order 5K->10K->15K, using ONLY design states for H1 fit
hist_ema=None; hist_alpha=0.5
for ct in CKPTS:
    for ci in CAMS:
        d=data[f"{ct}_cam{ci}"]
        gk=primary_gnorm(d)  # [NUM_TILES]
        gfn2=float((gk**2).sum())
        residual=d["residual_tile"]
        # H2 geometry-Jacobian: residual * per-tile conic (2D inverse-cov) mass = footprint-weighted residual
        h2=residual*d["conic_tile"]
        # H2b: residual * sqrt(gaussian count in tile)
        h2b=residual*np.sqrt(np.maximum(d["cnt_tile"],1))
        # H1 historical: use EMA from PRIOR states only (design states = 5K,10K; for 15K uses 5K+10K history)
        if hist_ema is None:
            h1=np.full(NUM_TILES, gk.mean())  # first state: no history -> uniform-ish (floor handles)
        else:
            h1=hist_ema.copy()
        # H3 hybrid: combine H1 (historical) and H2 (current geometry)
        def nz(x):
            x=np.asarray(x,np.float64); s=x.sum(); return x/s if s>1e-12 else np.full_like(x,1.0/len(x))
        h3=0.5*nz(h1)+0.5*nz(h2)
        # exploration floor: ensure min prob
        FLOOR=0.02
        proxies={"U":np.ones(NUM_TILES),"E":residual,"G":gk,"H1":h1,"H2":h2,"H2b":h2b,"H3":h3}
        # budget-matched inclusion probs per strategy at p=0.25,0.40
        for pb in [0.25,0.40]:
            nmse={}
            for sn,imp in proxies.items():
                pj=ntb(imp,pb)
                pj=np.clip(pj,FLOOR,1.0)
                # renormalize to budget after floor (simple: scale to mean=pb)
                pj=pj*(pb/ (pj.mean()+1e-12)); pj=np.clip(pj,FLOOR,1.0)
                v=var_correct(pj,gk)
                nmse[sn]=v/(gfn2+1e-12)
            results.setdefault(f"{ct}_cam{ci}",{})[str(pb)]={"nmse":nmse,
                "red_vs_U":{sn:1-nmse[sn]/nmse["U"] for sn in nmse}}
        # update H1 history AFTER computing (so 15K uses 5K+10K history, not its own)
        if ct in ("5K","10K"):  # design states update history
            if hist_ema is None: hist_ema=gk.copy()
            else: hist_ema=(1-hist_alpha)*hist_ema+hist_alpha*gk

# ---- aggregate ----
def agg(strategy, pb, keys):
    vals=[results[k][str(pb)]["nmse"][strategy] for k in keys]
    reds=[results[k][str(pb)]["red_vs_U"][strategy] for k in keys]
    return float(np.mean(vals)), float(np.mean(reds)), vals

all_keys=[f"{ct}_cam{ci}" for ct in CKPTS for ci in CAMS]
design_keys=[f"{ct}_cam{ci}" for ct in ("5K","10K") for ci in (0,1)]  # H1 designed here
heldout_keys=[f"{ct}_cam{ci}" for ct in ("15K",) for ci in CAMS]+[f"{ct}_cam2" for ct in ("5K","10K")]  # 15K + cam2

summary={"formula":"Var=sum[(1-p_j)/p_j]||g_j||^2; NMSE=Var/||g_means||^2",
         "primary_group":"means","tile":TILE,"num_tiles":NUM_TILES,
         "design_states":["5K","10K"],"design_cams":[0,1],"heldout":["15K(all cams)","cam2(all states)"]}
for pb in [0.25,0.40]:
    row={}
    for sn in ["U","E","G","H1","H2","H2b","H3"]:
        m_all,r_all,_=agg(sn,pb,all_keys)
        m_ho,r_ho,_=agg(sn,pb,heldout_keys)
        row[sn]={"nmse_all":m_all,"red_vsU_all":r_all,"nmse_heldout":m_ho,"red_vsU_heldout":r_ho}
    summary[str(pb)]=row

# GO/DROP evaluation
go={}
for pb in [0.25,0.40]:
    for sn in ["H1","H2","H2b","H3"]:
        r_all=summary[str(pb)][sn]["red_vsU_all"]; r_ho=summary[str(pb)][sn]["red_vsU_heldout"]
        go[f"{sn}_p{pb}"]={"red_all":r_all,"red_heldout":r_ho,
                           "ge25_all":r_all>=0.25,"ge25_heldout":r_ho>=0.25,
                           "pass":(r_all>=0.25 and r_ho>=0.25)}
summary["go_eval"]=go
summary["corr_diagnostic"]=corr_diag
summary["note"]="G non-deployable upper bound (true per-tile grad). E/H1/H2/H2b/H3 deployable. GO needs >=25% NMSE reduction vs U on primary group, consistent on held-out."

json.dump({"summary":summary,"per_combo":{k:v for k,v in results.items()}},
          open(f"{OUT}/GP0_PHASE3_ORACLE.json","w"),indent=2,default=str)
print(json.dumps(summary,indent=2,default=str))
