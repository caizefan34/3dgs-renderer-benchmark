#!/usr/bin/env python3
"""Aggregate G3 gpu microbench JSONs -> G3_GPU_MICROBENCH.json (4 scenes, fixed teacher)."""
import json, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
REC=os.path.join(HERE,"G1_RAW_RECORDS")
RANKS=(4,8,16)
SCENES=[("bicycle","B1_bicycle_30k/checkpoints/iter_30000.pt","bicycle_r30k"),
        ("room","B1_room_multiphase_v2/checkpoints/iter_5000.pt","room_r5k"),
        ("room","B1_room_30k/checkpoints/iter_30000.pt","room_r30k"),
        ("garden","B1_garden_30k/checkpoints/iter_30000.pt","garden_r30k")]
def main():
    per={}; intrinsic={}; ptile={}; Smeas={}; hover={}; host_note=None; Mrep={}
    for scene,ckrel,tag in SCENES:
        fp=os.path.join(REC,f"g3_{tag}.json")
        if not os.path.exists(fp):
            print("MISSING",fp); continue
        d=json.load(open(fp))
        host_note=d.get("host",host_note)
        hl=d.get("headline",{})
        per[tag]={"scene":scene,"ckpt":"/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/"+ckrel,
                  "n_tiles":d["n_tiles"],"M_median":d["M_median"],"M_min":d["M_min"],"M_max":d["M_max"],
                  "dispatch_overhead_ms":d["dispatch_overhead_ms"],
                  "intrinsic_gamma":{f"R{r}":hl.get(f"intrinsic_gamma_R{r}") for r in RANKS},
                  "per_tile_gamma":{f"R{r}":d["per_rank"][str(r)]["gamma_work_weighted"] for r in RANKS},
                  "h_overhead_per_tile":d["per_rank"]["4"]["h_overhead_per_tile"],
                  "S_measured_zero_idealh":{f"R{r}":d["per_rank"][str(r)]["S_measured_zero_idealh"] for r in RANKS},
                  "S_measured_with_h":{f"R{r}":d["per_rank"][str(r)]["S_measured_with_h"] for r in RANKS}}
        for r in RANKS:
            intrinsic.setdefault(f"R{r}",[]).append(hl.get(f"intrinsic_gamma_R{r}"))
            ptile.setdefault(f"R{r}",[]).append(d["per_rank"][str(r)]["gamma_work_weighted"])
            Smeas.setdefault(f"R{r}",[]).append(d["per_rank"][str(r)]["S_measured_with_h"])
        hover[tag]=d["per_rank"]["4"]["h_overhead_per_tile"]
    def rng(v):
        v=[x for x in v if x is not None]
        return {"min":float(min(v)),"median":float(np.median(v)),"max":float(max(v))} if v else {}
    agg_intrinsic={k:rng(v) for k,v in intrinsic.items()}
    agg_ptile={k:rng(v) for k,v in ptile.items()}
    agg_S={k:rng(v) for k,v in Smeas.items()}
    hvals=[v for v in hover.values()]
    out={"gate":"G3_gpu_local_microbench",
         "host":host_note,
         "method":"Reuse G1 gsplat packed tile extraction; port teacher (M-Gaussian alpha composite) and student (rank-R CGT mixture) to torch GPU; CUDA-event timing, warmup+median. Two regimes: per-tile (256px, launch-bound) and intrinsic (65536px single call, launch-free math ratio). Params created once; no H2D in timing loops. Teacher/student port verified against G1 numpy reference.",
         "scenes_run":[s[2] for s in SCENES],
         "scenes_pending_gpu_available":[],
         "pending_reason":"",
         "per_scene":per,
         "aggregate_intrinsic_gamma":agg_intrinsic,
         "aggregate_per_tile_gamma":agg_ptile,
         "aggregate_S_measured_with_h":agg_S,
         "aggregate_h_overhead_per_tile":{"min":float(min(hvals)),"median":float(np.median(hvals)),"max":float(max(hvals))},
         "findings":[
           "Intrinsic (launch-free) per-element math gamma ~ M_rep/R (representative-tile Gaussian count / rank), ranging ~1.3x (room, low-M) to ~78x (bicycle, high-M). This is the MATH lower bound, NOT achievable by a per-tile renderer.",
           "Per-tile (256px, launch-bound) gamma collapses to ~1.4-2.4x across scenes/ranks; measured per-tile speedup with dispatch overhead S~1.2-2.2x (rank-4 best), far below the plan 3x target.",
           "Higher rank does NOT monotonically improve per-tile S (non-monotonic), consistent with G1 Jacobian non-monotonicity."],
         "verdict":"G3_PARTIAL_MEASURED_NOT_A_3X_SPEEDUP: per-tile launch-bound overhead caps speedup well below 3x; recovering headroom needs batching/fusion (G4) and a full spatial-tree cost model (not implemented).",
         "scope_note":"G3 measures a LOCAL per-tile kernel ratio, NOT an end-to-end renderer speedup. NOT reported as a speedup claim."}
    json.dump(out,open(os.path.join(HERE,"G3_GPU_MICROBENCH.json"),"w"),indent=1)
    print("g3 agg OK; scenes:",list(per.keys()))
    print("intrinsic_gamma:",{k:round(v["median"],1) for k,v in agg_intrinsic.items()})
    print("per_tile_gamma:",{k:round(v["median"],2) for k,v in agg_ptile.items()})
    print("S_with_h:",{k:round(v["median"],2) for k,v in agg_S.items()})
    print("h_overhead:",round(out["aggregate_h_overhead_per_tile"]["median"],3))
if __name__=="__main__": main()
