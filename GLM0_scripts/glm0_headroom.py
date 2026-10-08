#!/usr/bin/env python3
"""GLM0 Phase 3: Lifecycle Headroom Oracle (no real skip; pure accounting).
3-stage lifecycle per uid (transparent rule on the two discovered signals):
  INACTIVE : vis_frac < TV           (low contribution -> skip, w=1)
  NEW      : vis_frac >= TV & age < TA (young, ramping -> full update, w=0)
  MATURE   : vis_frac >= TV & age >= TA (established+contributing -> reduce, w=1-1/freq)
On frozen-C0 sampled iterations:
  Baseline : every gaussian updated every iter
  VOS      : skip updates for invisible gaussians
  GLM      : VOS + lifecycle skip/reduce
additional_safe_skip_headroom = (GLM_skip - VOS_skip) / baseline
Primary TV=0.05, TA=3000; sensitivity over TV, TA, freq. Pure metadata arithmetic.
"""
import os, numpy as np, pandas as pd, json

RUNS={"room":"/root/epic05_d0/runs/TRACE_room",
      "bicycle":"/root/epic05_d0/runs/TRACE_bicycle",
      "garden":"/root/glm0/runs/TRACE_garden"}
OUT="/root/glm0/work"; ART="/root/glm0/artifacts"

def stage_of(uid, TV, TA):
    vf=uid.vis_frac.values; age=uid.age.values
    st=np.where(vf<TV,"INACTIVE",np.where(age<TA,"NEW","MATURE"))
    return st

def scene_oracle(scene, TV, TA, freq):
    ss=np.load(os.path.join(RUNS[scene],"d0_sample_summary.npz"))
    Nt=ss["N_t"].astype(np.float64); nvis=ss["n_visible"].astype(np.float64)
    baseline=float(Nt.sum()); vos_skip=float((Nt-nvis).sum()); visible=float(nvis.sum())
    uid=pd.read_parquet(os.path.join(OUT,f"uidtable_{scene}.parquet"))
    st=stage_of(uid,TV,TA)
    vis_count=uid.vis_count.values.astype(np.float64)
    w=np.zeros(st.shape[0],dtype=np.float64)
    w[st=="INACTIVE"]=1.0
    w[st=="MATURE"]=1.0-1.0/freq
    additional=float((w*vis_count).sum())
    glm_skip=vos_skip+additional
    inact=st=="INACTIVE"
    return {"scene":scene,"TV":TV,"TA":TA,"freq":freq,
      "baseline":baseline,"vos_skip":vos_skip,"visible":visible,"glm_skip":glm_skip,
      "additional":additional,
      "vos_headroom":vos_skip/baseline,"glm_headroom":glm_skip/baseline,
      "additional_safe_skip_headroom":additional/baseline,
      "additional_over_visible":additional/visible if visible>0 else 0.0,
      "stage_frac":{s:float((st==s).mean()) for s in ["NEW","MATURE","INACTIVE"]},
      "stage_visrec":{s:float(vis_count[st==s].sum()) for s in ["NEW","MATURE","INACTIVE"]},
      "inactive_always_invis_frac":float((uid.vis_count.values[inact]==0).mean()) if inact.sum() else 0.0}

if __name__=="__main__":
    import sys
    scenes=sys.argv[1:] or list(RUNS)
    res={"scenes":{},"sensitivity":{},"pooled":{},"gate":{}}
    # primary config
    TV,TA,freq=0.05,3000,2
    per=[]
    for s in scenes:
        r=scene_oracle(s,TV,TA,freq); per.append(r); res["scenes"][s]=r
    agg={k:sum(r[k] for r in per) for k in ["baseline","vos_skip","visible","additional"]}
    res["pooled"]={"primary":{**{k:agg[k] for k in agg},
      "vos_headroom":agg["vos_skip"]/agg["baseline"],
      "glm_headroom":(agg["vos_skip"]+agg["additional"])/agg["baseline"],
      "additional_safe_skip_headroom":agg["additional"]/agg["baseline"],
      "additional_over_visible":agg["additional"]/agg["visible"]}}
    # sensitivity: freq (primary TV,TA)
    for f in (1,2,4):
        pf=[scene_oracle(s,TV,TA,f) for s in scenes]
        a=sum(r["additional"] for r in pf); b=sum(r["baseline"] for r in pf)
        res["sensitivity"][f"freq{f}_TV{TV}_TA{TA}"]={"additional_safe_skip_headroom":a/b,
            "glm_headroom":(sum(r["vos_skip"] for r in pf)+a)/b}
    # sensitivity: TV (primary TA,freq)
    for tv in (0.03,0.05,0.10,0.15):
        pf=[scene_oracle(s,tv,TA,freq) for s in scenes]
        a=sum(r["additional"] for r in pf); b=sum(r["baseline"] for r in pf)
        res["sensitivity"][f"TV{tv}_TA{TA}_freq{freq}"]={"additional_safe_skip_headroom":a/b,
            "glm_headroom":(sum(r["vos_skip"] for r in pf)+a)/b}
    # sensitivity: TA (primary TV,freq)
    for ta in (1500,3000,5000):
        pf=[scene_oracle(s,TV,ta,freq) for s in scenes]
        a=sum(r["additional"] for r in pf); b=sum(r["baseline"] for r in pf)
        res["sensitivity"][f"TA{ta}_TV{TV}_freq{freq}"]={"additional_safe_skip_headroom":a/b,
            "glm_headroom":(sum(r["vos_skip"] for r in pf)+a)/b}
    p=res["pooled"]["primary"]
    res["gate"]={"primary_additional_safe_skip_headroom":p["additional_safe_skip_headroom"],
                 "gate_ge_10pct":bool(p["additional_safe_skip_headroom"]>=0.10),
                 "config":"TV=0.05,TA=3000,freq=2",
                 "freq4_additional":res["sensitivity"]["freq4_TV0.05_TA3000"]["additional_safe_skip_headroom"]}
    json.dump(res,open(os.path.join(ART,"GLM0_HEADROOM.json"),"w"),indent=2)
    print("PRIMARY (TV=0.05,TA=3000,freq=2):")
    for r in per:
        print(f"  {r['scene']}: VOS={r['vos_headroom']:.4f} GLM={r['glm_headroom']:.4f} add={r['additional_safe_skip_headroom']:.4f} add/vis={r['additional_over_visible']:.4f} stages={ {k:round(v,2) for k,v in r['stage_frac'].items()} }")
    print(f'POOLED: VOS={p['vos_headroom']:.4f} GLM={p['glm_headroom']:.4f} additional={p['additional_safe_skip_headroom']:.4f} add/vis={p['additional_over_visible']:.4f}')
    print('sensitivity additional_safe_skip_headroom:', {k:round(v['additional_safe_skip_headroom'],4) for k,v in res['sensitivity'].items()})
    print('GATE>=10%:',res['gate']['gate_ge_10pct'])
