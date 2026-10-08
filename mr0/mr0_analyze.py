
#!/usr/bin/env python3
import json,sys
CELLS=("room@10000","room@15000","bicycle@10000","bicycle@15000")
KS=("4","8","16")
def strong(x):
    return (x["train_psnr_delta_median_db"]>=-.08 and x["truehold_psnr_delta_median_db"]>=-.08 and
            x["truehold_psnr_delta_worst_db"]>=-.20 and x["ssim_delta_median"]>=-.002 and
            x["response_cos_median"]>=.90 and x["response_cos_p10"]>=.80 and
            x["intersection_reduction"]>=.65 and x["optimized_variable_reduction"]>=.65 and
            x["predicted_iteration_speedup"]>=3.0 and x["nan_inf_count"]==0)
def promote(x):
    return (x["train_psnr_delta_median_db"]>=-.15 and x["truehold_psnr_delta_median_db"]>=-.15 and
            x["truehold_psnr_delta_worst_db"]>=-.35 and x["ssim_delta_median"]>=-.004 and
            x["response_cos_median"]>=.82 and x["response_cos_p10"]>=.68 and
            x["intersection_reduction"]>=.55 and x["optimized_variable_reduction"]>=.55 and
            x["predicted_iteration_speedup"]>=3.0 and x["nan_inf_count"]==0)
d=json.load(open(sys.argv[1]))
if d.get("requires_anchor_family_features",False):
    out={"verdict":"MR0_NOT_DISTINCT_FROM_ANCHOR_FAMILY","selected_K":None}
else:
    ss=[]; pp=[]
    for k in KS:
        xs=[d["cells"][c]["K"][k]["coverage75"] for c in CELLS]
        if all(strong(x) for x in xs): ss.append(k)
        if all(promote(x) for x in xs): pp.append(k)
    if ss: v="MR0_STRONG_PROMOTE"; sel=ss[0]
    elif pp: v="MR0_PROMOTE"; sel=pp[0]
    else: v="MR0_DROP"; sel=None
    out={"verdict":v,"selected_K":sel,"strong_K":ss,"promote_K":pp}
open(sys.argv[1].replace("SUMMARY","VERDICT"),"w").write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
