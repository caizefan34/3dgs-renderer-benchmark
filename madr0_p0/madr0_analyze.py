#!/usr/bin/env python3
import json,sys
CELLS=("room@10000","room@15000","bicycle@10000","bicycle@15000")
d=json.load(open(sys.argv[1]))

if not d.get("dpc_base_semantics_pass",False):
    out={"verdict":"MADR0_BLOCKED_DPC_BASE"}
elif not d.get("same_gpu_determinism_pass",False):
    out={"verdict":"MADR0_NONDETERMINISTIC"}
elif not d.get("gradient_correctness_pass",False) or not d.get("one_step_pass",False):
    out={"verdict":"MADR0_GRADIENT_INCORRECT"}
else:
    def promote(x):
        return (x["same_gpu_deterministic"] and
                x["cos_vs_dpc"]>=.9999999 and
                x["rel_l2_vs_dpc"]<=1e-5 and
                x["reduction_speedup_vs_dpc"]>=1.25 and
                x["total_backward_speedup_vs_dpc"]>=1.10 and
                x["extra_vram_fraction"]<=.25 and
                x["nan_inf_count"]==0)
    def strong(x):
        return (promote(x) and x["cross_a100_bitwise_equal"] and
                x["reduction_speedup_vs_dpc"]>=1.50 and
                x["total_backward_speedup_vs_dpc"]>=1.20 and
                x["absolute_speedup_vs_c0"]>=1.75 and
                x["extra_vram_fraction"]<=.15)
    xs=[d["cells"][c] for c in CELLS]
    p=all(promote(x) for x in xs)
    s=all(strong(x) for x in xs)
    med=float(d["global"]["total_backward_speedup_vs_dpc_median"])
    if s and med>=1.30: v="MADR0_STRONG_PROMOTE"
    elif p and med>=1.15: v="MADR0_PROMOTE"
    else: v="MADR0_TOO_SLOW"
    out={"verdict":v,"promote_all_cells":p,"strong_all_cells":s,
         "median_total_gain_vs_dpc":med}
open(sys.argv[1].replace("SUMMARY","VERDICT"),"w").write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
