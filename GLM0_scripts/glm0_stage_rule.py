#!/usr/bin/env python3
"""GLM0 Phase 2 (operational): transparent 3-stage lifecycle labeling on the
discovered signals, consistent with the headroom oracle. Adds rule_three_stage
to GLM0_STAGE_DISCOVERY.json. GMM (BIC k=4-5) remains as unsupervised evidence.
  INACTIVE: vis_frac < 0.05 ; NEW: vis_frac>=0.05 & age<3000 ; MATURE: else
"""
import os, numpy as np, pandas as pd, json
OUT="/root/glm0/work"; ART="/root/glm0/artifacts"; TV=0.05; TA=3000
def vlevel(x): return 0 if x<0.05 else (1 if x<0.25 else 2)
res=json.load(open(os.path.join(ART,"GLM0_STAGE_DISCOVERY.json")))
for scene in res:
    df=pd.read_parquet(os.path.join(OUT,f"uidtable_{scene}.parquet"))
    vf=df.vis_frac.values; age=df.age.values
    st=np.where(vf<TV,"INACTIVE",np.where(age<TA,"NEW","MATURE"))
    sn=pd.Series(st)
    stage_num=sn.map({"NEW":0,"MATURE":2,"INACTIVE":-1}).values
    vis=df.vis_frac.values; grad=np.nan_to_num(df.grad_mean_norm.values,nan=0.0)
    def corr(a,b):
        m=np.isfinite(a)&np.isfinite(b); return float(np.corrcoef(a[m],b[m])[0,1])
    # early/late visibility transition
    ve=pd.Series(np.where(df.vis_frac_early.values<0.05,0,np.where(df.vis_frac_early.values<0.25,1,2)))
    vl=pd.Series(np.where(df.vis_frac_late.values<0.05,0,np.where(df.vis_frac_late.values<0.25,1,2)))
    tm2=pd.crosstab(ve,vl,normalize="index").values
    lt={}
    for s in ["NEW","MATURE","INACTIVE"]:
        m=st==s; sub=df[m]
        lt[s]={"n":int(m.sum()),"frac":float(m.mean()),"age_median":float(sub.age.median()),
               "vis_frac_median":float(sub.vis_frac.median()),"tiles_cum_median":float(sub.tiles_cum.median()),
               "alive_frac":float(sub.alive.mean()),"grad_median":float(sub.grad_mean_norm.median())}
    res[scene]["rule_three_stage"]={"TV":TV,"TA":TA,"definition":"INACTIVE:vis_frac<0.05; NEW:vis_frac>=0.05 & age<3000; MATURE:vis_frac>=0.05 & age>=3000",
      "stage_distribution":{k:float(v) for k,v in sn.value_counts(normalize=True).items()},
      "lifetime_statistics":lt,
      "visibility_relation":{"vis_frac_by_stage":{s:float(df[st==s].vis_frac.median()) for s in ["NEW","MATURE","INACTIVE"]},
                            "corr_stage_vs_vis_frac":corr(stage_num,vis)},
      "gradient_relation":{"grad_median_by_stage":{s:float(df[st==s].grad_mean_norm.median()) for s in ["NEW","MATURE","INACTIVE"]},
                           "corr_stage_vs_grad":corr(stage_num,grad)},
      "transition_matrix_early_to_late":{"levels":["low","mid","high"],"matrix":tm2.tolist()}}
    print(scene,"rule3:",{k:round(v,3) for k,v in res[scene]["rule_three_stage"]["stage_distribution"].items()})
json.dump(res,open(os.path.join(ART,"GLM0_STAGE_DISCOVERY.json"),"w"),indent=2)
print("updated GLM0_STAGE_DISCOVERY.json with rule_three_stage")
