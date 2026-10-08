#!/usr/bin/env python3
"""GLM0 Phase 2: Lifecycle Stage Discovery (unsupervised).
GMM(diag) on standardized lifetime features. Reports BIC sweep (fine structure,
kmin..kmax) AND a k=3 three-stage labeling (goal granularity NEW/MATURE/INACTIVE)
used by the headroom oracle. Clusters labeled by centroid.
"""
import os, numpy as np, pandas as pd, json, sys
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

OUT="/root/glm0/work"; ART="/root/glm0/artifacts"; T_END=30000

def load(scene):
    df=pd.read_parquet(os.path.join(OUT,f"uidtable_{scene}.parquet"))
    X=pd.DataFrame(index=df.index)
    X["f_age"]=np.log1p(df.age.values)
    X["f_vis"]=df.vis_frac.values
    X["f_contrib"]=np.log1p(df.tiles_cum.values)
    X["f_op"]=df.parent_opacity.values.astype(float)
    X=X.fillna(X.median(numeric_only=True))
    return df,X

def c3_profile(prof3):
    return prof3.groupby("label").agg(n=("age","size"),age=("age","median"),vis=("vis_frac","median"),
        tiles=("tiles_cum","median"),alive_frac=("alive","mean"),
        vis_early=("vis_frac_early","median"),vis_late=("vis_frac_late","median"))

def discover(scene,kmin=2,kmax=5,subn=250000,seed=42):
    df,X=load(scene); Xv=X.values.astype(np.float64)
    rng=np.random.RandomState(seed)
    idx=rng.choice(len(Xv),size=min(subn,len(Xv)),replace=False)
    sc=StandardScaler().fit(Xv[idx]); Zsub=sc.transform(Xv[idx])
    bic={}
    for k in range(kmin,kmax+1):
        gm=GaussianMixture(n_components=k,covariance_type="diag",random_state=seed,n_init=2,max_iter=200).fit(Zsub)
        bic[k]=float(gm.bic(Zsub))
    kbest=min(bic,key=bic.get)
    # fine labeling
    gm=GaussianMixture(n_components=kbest,covariance_type="diag",random_state=seed,n_init=4,max_iter=300).fit(Zsub)
    lab=gm.predict(sc.transform(Xv))
    prof=pd.DataFrame({"label":lab,"age":df.age.values,"vis_frac":df.vis_frac.values,
        "tiles_cum":df.tiles_cum.values,"vis_count":df.vis_count.values,
        "grad":np.nan_to_num(df.grad_mean_norm.values,nan=0.0),"alive":df.alive.values,
        "vis_frac_early":df.vis_frac_early.values,"vis_frac_late":df.vis_frac_late.values})
    cent=c3_profile(prof); cent["frac"]=cent.n/len(prof)
    # k=3 three-stage labeling (goal granularity)
    gm3=GaussianMixture(n_components=3,covariance_type="diag",random_state=seed,n_init=4,max_iter=300).fit(Zsub)
    lab3=gm3.predict(sc.transform(Xv))
    prof3=pd.DataFrame({"label":lab3,"age":df.age.values,"vis_frac":df.vis_frac.values,
        "tiles_cum":df.tiles_cum.values,"vis_count":df.vis_count.values,
        "grad":np.nan_to_num(df.grad_mean_norm.values,nan=0.0),"alive":df.alive.values,
        "vis_frac_early":df.vis_frac_early.values,"vis_frac_late":df.vis_frac_late.values})
    c3=c3_profile(prof3); c3["frac"]=c3.n/len(prof3)
    c3s=c3.sort_values(["vis","tiles"],ascending=False)
    inact_lab=int(c3s.index[-1]); mat_lab=int(c3s.index[0])
    rem=[int(i) for i in c3s.index if i not in (inact_lab,mat_lab)]
    new_lab=min(rem,key=lambda i:c3.loc[i,"age"]) if rem else mat_lab
    m3={inact_lab:"INACTIVE",mat_lab:"MATURE",new_lab:"NEW"}
    prof3["stage"]=prof3.label.map(m3)
    # fine labels
    mat_f=cent.sort_values(["vis","tiles"],ascending=False).index[0]
    ina_f=cent.sort_values(["vis","tiles"],ascending=True).index[0]
    lmap={}
    for l in cent.index:
        l=int(l)
        if l==mat_f: lmap[l]="MATURE"
        elif l==ina_f: lmap[l]="INACTIVE"
        else: lmap[l]="NEW" if cent.loc[l,"age"]<cent.age.median() else "MID"
    prof["stage"]=prof.label.map(lmap)
    prof.to_parquet(os.path.join(OUT,f"prof_{scene}.parquet"),index=False)
    prof3.to_parquet(os.path.join(OUT,f"prof3_{scene}.parquet"),index=False)
    three={"k":3,"labels":{str(k):v for k,v in m3.items()},
      "profile":{str(i):{kk:(float(vv) if kk!="n" else int(vv)) for kk,vv in r.items()} for i,r in c3.iterrows()},
      "stage_distribution":{k:float(v) for k,v in prof3.stage.value_counts(normalize=True).items()}}
    fine={"k":kbest,"labels":{str(k):v for k,v in lmap.items()},
      "profile":{str(i):{kk:(float(vv) if kk!="n" else int(vv)) for kk,vv in r.items()} for i,r in cent.iterrows()},
      "stage_distribution":{k:float(v) for k,v in prof.stage.value_counts(normalize=True).items()}}
    return df,prof,prof3,cent,c3,fine,three,bic,kbest

def vlevel(x): return 0 if x<0.05 else (1 if x<0.25 else 2)

def rel_stats(stage_arr, df, prefix):
    us=sorted(set(stage_arr))
    sn=pd.Series(stage_arr)
    stage_num=sn.map({"NEW":0,"MID":1,"MATURE":2,"INACTIVE":-1}).values
    vis=df.vis_frac.values; grad=np.nan_to_num(df.grad_mean_norm.values,nan=0.0)
    def corr(a,b):
        msk=np.isfinite(a)&np.isfinite(b); return float(np.corrcoef(a[msk],b[msk])[0,1]) if msk.sum()>2 else None
    return {"vis_frac_by_stage":{s:float(df[sn.values==s].vis_frac.median()) for s in us},
            "grad_median_by_stage":{s:float(df[sn.values==s].grad_mean_norm.median()) for s in us},
            "corr_stage_vs_vis_frac":corr(stage_num,vis),
            "corr_stage_vs_grad":corr(stage_num,grad)}

def summarize(scene,df,prof,prof3,cent,c3,fine,three,kbic,kbest):
    out={"scene":scene,"bic":{str(k):v for k,v in kbic.items()},"kbest":kbest,
         "fine_stages":fine,"three_stage":three}
    for tag,pa in [("fine",prof),("three",prof3)]:
        tm=pd.crosstab(pa.vis_frac_early.map(vlevel),pa.vis_frac_late.map(vlevel),normalize="index").values
        lt={}
        for s in ["NEW","MID","MATURE","INACTIVE"]:
            m=pa.stage.values==s
            if m.sum()==0: continue
            sub=df[m]
            lt[s]={"n":int(m.sum()),"age_median":float(sub.age.median()),
                   "age_p10":float(sub.age.quantile(.1)),"age_p90":float(sub.age.quantile(.9)),
                   "vis_frac_median":float(sub.vis_frac.median()),
                   "tiles_cum_median":float(sub.tiles_cum.median()),"alive_frac":float(sub.alive.mean())}
        out[tag]={"transition_matrix_early_to_late":{"levels":["low(<0.05)","mid(0.05-0.25)","high(>0.25)"],"matrix":tm.tolist()},
                  "lifetime_statistics":lt, **rel_stats(pa.stage.values,df,tag)}
    return out

if __name__=="__main__":
    scenes=sys.argv[1:] or ["room","bicycle"]; res={}
    for s in scenes:
        df,prof,prof3,cent,c3,fine,three,kbic,kbest=discover(s)
        res[s]=summarize(s,df,prof,prof3,cent,c3,fine,three,kbic,kbest)
        ts=res[s]["three_stage"]["stage_distribution"]
        print(f"[{s}] kbest={kbest} three_stage="+str({k:round(v,3) for k,v in ts.items()}))
    json.dump(res,open(os.path.join(ART,"GLM0_STAGE_DISCOVERY.json"),"w"),indent=2)
    print("WROTE GLM0_STAGE_DISCOVERY.json")
