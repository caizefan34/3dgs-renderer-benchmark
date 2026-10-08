#!/usr/bin/env python3
"""GLM0 Phase 1: Gaussian Lifecycle Trace -> GLM0_TRACE.json.
Per scene: population trajectory (N_t sampled), lifecycle events (births by
type, deaths by reason), per-Gaussian statistics (lifetime histogram, survival
curve, surviving ratio, activity distributions). Metadata-only, from frozen-C0
traces. Pure read of uidtables + population trajectories.
"""
import os, numpy as np, pandas as pd, json
RUNS={"room":"/root/epic05_d0/runs/TRACE_room","bicycle":"/root/epic05_d0/runs/TRACE_bicycle","garden":"/root/glm0/runs/TRACE_garden"}
OUT="/root/glm0/work"; ART="/root/glm0/artifacts"; T_END=30000
def hist(vals,bins):
    h,_=np.histogram(vals,bins=bins); return {"bins":list(bins),"counts":h.tolist()}
def scene_trace(scene):
    df=pd.read_parquet(os.path.join(OUT,f"uidtable_{scene}.parquet"))
    ss=np.load(os.path.join(RUNS[scene],"d0_sample_summary.npz"))
    Nt=ss["N_t"].astype(np.int64); iters=ss["iter"].astype(np.int64)
    births={"init":int((df.birth_type==0).sum()),"clone":int((df.birth_type==1).sum()),
            "split":int((df.birth_type==2).sum())}
    deaths={"opacity":int((df.terminal_reason==1).sum()),
            "screen":int((df.terminal_reason==2).sum()),
            "world":int((df.terminal_reason==3).sum()),
            "split_parent":int((df.terminal_reason==4).sum())}
    lifetime=df.age.values.astype(float); lifetime[df.alive.values==1]=np.nan
    lt_valid=lifetime[~np.isnan(lifetime)]
    # survival curve: fraction of cohort alive at age a (per birth cohort, using all uids)
    ages=np.sort(df.age.values)
    surv_a=np.arange(1,len(ages)+1)/len(ages)  # cumulative (age is lifetime; alive have age=30001-birth)
    # sample survival curve at age thresholds
    sc_points=[]
    for a in [0,1000,2000,5000,10000,15000,20000,25000,30000]:
        sc_points.append({"age":int(a),"surviving_frac":float((df.age.values>a).mean())})
    activity={"vis_frac":{ "mean":float(df.vis_frac.mean()),"median":float(df.vis_frac.median()),
                           "p10":float(df.vis_frac.quantile(.1)),"p90":float(df.vis_frac.quantile(.9))},
              "tiles_cum":{"mean":float(df.tiles_cum.mean()),"median":float(df.tiles_cum.median()),
                           "p10":float(df.tiles_cum.quantile(.1)),"p90":float(df.tiles_cum.quantile(.9))},
              "grad_mean_norm":{"median":float(df.grad_mean_norm.median()),
                                "p90":float(df.grad_mean_norm.quantile(.9))}}
    return {"scene":scene,"iterations":int(T_END),"sampled_iter_count":int(len(iters)),
      "gaussian_count":{"N_final":int(Nt[-1]),"N_max":int(Nt.max()),"N_init":int(Nt[0]),
        "N_trajectory_sampled":{"iter":iters.tolist()[:50],"N":Nt.tolist()[:50],
          "note":"first 50 sampled; full trajectory in d0_population_trajectory.npz"}},
      "lifecycle_events":{"births":births,"deaths":deaths,
        "birth_total":int(sum(births.values())),"death_total":int(sum(deaths.values())),
        "alive_at_end":int(df.alive.sum())},
      "per_gaussian_statistics":{
        "lifetime_iters":{"n_with_death":int(len(lt_valid)),"mean":float(lt_valid.mean()) if len(lt_valid) else 0,
          "median":float(np.median(lt_valid)) if len(lt_valid) else 0,
          "p10":float(np.percentile(lt_valid,10)) if len(lt_valid) else 0,
          "p90":float(np.percentile(lt_valid,90)) if len(lt_valid) else 0,
          "histogram":hist(lt_valid,[0,1000,2000,5000,10000,15000,20000,25000,30000,31000])},
        "surviving_ratio":{"alive_frac_at_30k":float(df.alive.mean()),
          "curve":sc_points},
        "activity_distributions":activity}}
if __name__=="__main__":
    import sys
    scenes=sys.argv[1:] or list(RUNS)
    res={"phases":"GLM0 Phase 1 — Gaussian Lifecycle Trace (frozen C0, seed 42, 30K)",
         "scenes":{}}
    for s in scenes:
        res["scenes"][s]=scene_trace(s)
        print(f"[{s}] N_init={res['scenes'][s]['gaussian_count']['N_init']} N_final={res['scenes'][s]['gaussian_count']['N_final']} births={res['scenes'][s]['lifecycle_events']['birth_total']} deaths={res['scenes'][s]['lifecycle_events']['death_total']} alive={res['scenes'][s]['lifecycle_events']['alive_at_end']}")
    json.dump(res,open(os.path.join(ART,"GLM0_TRACE.json"),"w"),indent=2)
    print("WROTE GLM0_TRACE.json")
