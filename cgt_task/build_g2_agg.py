#!/usr/bin/env python3
"""Rebuild G2_CONTINUOUS_DOMAIN_CERTIFICATE.json (dense aggregate, pooled across scenes)."""
import json, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
CERT=os.path.join(HERE,"G2_CERT")
RANKS=(4,8,16); GATE=0.02
SCENE_ORDER=[("room","B1_room_multiphase_v2/checkpoints/iter_15000.pt","room_r15k"),
             ("garden","B1_garden_30k/checkpoints/iter_30000.pt","garden_r30k"),
             ("bicycle","B1_bicycle_30k/checkpoints/iter_30000.pt","bicycle_r30k")]
def pctl(a,q):
    a=np.asarray(a,dtype=float); return float(np.percentile(a,q)) if len(a) else 0.0
def main():
    dense_per={}; pool={r:{"mx":[],"ho":[],"M":[]} for r in RANKS}; mono={}; worst=[]
    for scene,ckrel,tag in SCENE_ORDER:
        d=json.load(open(os.path.join(CERT,f"G2_CONTINUOUS_DOMAIN_CERTIFICATE_{tag}.json")))
        pr=d["per_rank"]; recs=d["records"]
        Mmed=float(np.median([x["M"] for x in recs if int(x["rank"])==4])) if recs else 0.0
        entry={"ckpt":"/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/"+ckrel,
               "n_tiles":d["n_tiles"],"sub":d["sub"],"n_dense_grid":d["n_dense_grid"],
               "M_median":Mmed,"per_rank":{}}
        prb=[]
        for r in RANKS:
            entry["per_rank"][str(r)]=pr[str(r)]
            prb.append(pr[str(r)].get("dense_pass_rate",0.0))
            for x in recs:
                if int(x["rank"])==r:
                    pool[r]["mx"].append(x["dense_max_abs"]); pool[r]["ho"].append(x["heldout_max_abs"]); pool[r]["M"].append(x["M"])
                    if x["dense_max_abs"]>GATE:
                        worst.append({"scene":scene,"rank":r,"tile_idx":x["tile_idx"],"M":x["M"],"dense_max_abs":x["dense_max_abs"]})
        dense_per[scene]=entry
        mono[scene]={"dense_pass_rate_by_rank":prb,
                     "non_monotonic":not all(prb[i]>=prb[i+1]-1e-9 for i in range(len(prb)-1))}
    pooled={}
    for r in RANKS:
        mx=np.asarray(pool[r]["mx"]); ho=np.asarray(pool[r]["ho"]); M=np.asarray(pool[r]["M"])
        pooled[str(r)]={"n":int(len(mx)),
            "dense_max_abs_median":pctl(mx,50),"dense_max_abs_p95":pctl(mx,95),
            "dense_max_abs_max":float(mx.max()) if len(mx) else 0.0,
            "dense_pass_rate":float((mx<=GATE).mean()) if len(mx) else 0.0,
            "heldout_pass_rate":float((ho<=GATE).mean()) if len(ho) else 0.0,
            "dense_vs_heldout_ratio_median":float(np.median(mx/(ho+1e-12))) if len(mx) else 0.0,
            "M_median":float(np.median(M)) if len(M) else 0.0}
    worst=sorted(worst,key=lambda x:-x["dense_max_abs"])[:8]
    out={"gate":"G2_continuous_domain_certificate_DENSE_COVERAGE_MULTI_SCENE",
         "method":"Deterministic dense sub-pixel grid coverage per tile (sub=4 -> 4096 pts over tile ray extent); fit student on even pixels (as G1), eval teacher vs student max|E,T| on dense grid and on held-out odd pixels.",
         "value_gate_max_abs":GATE,"ranks":list(RANKS),
         "scope":"EMPIRICAL dense-coverage certificate: max error over a deterministic dense sub-pixel grid. STRICTLY stronger than random held-out (deterministic+densely covers tile extent) but still a SAMPLE, NOT a rigorous interval-arithmetic bound over the continuum (rigorous interval-arithmetic result in G2_RIGOROUS_CERTIFICATE.json).",
         "honesty":"Dense coverage exposes a hard-tile TAIL that random held-out partly masked: worst-tile max|E,T| exceeds gate (0.02). Higher rank does NOT monotonically improve dense coverage, consistent with G1 Jacobian non-monotonicity. This is NOT the G1 gate (which is work-weighted compressibility); it is an add-on continuous-domain probe that TEMPERS any 'student reproduces teacher everywhere' claim.",
         "per_scene":dense_per,"pooled_all_scenes":pooled,"monotonicity":mono,"worst_tiles":worst}
    json.dump(out,open(os.path.join(HERE,"G2_CONTINUOUS_DOMAIN_CERTIFICATE.json"),"w"),indent=1)
    print("dense agg OK; pooled pass_rate:",{r:round(pooled[str(r)]["dense_pass_rate"],3) for r in RANKS})
if __name__=="__main__": main()
