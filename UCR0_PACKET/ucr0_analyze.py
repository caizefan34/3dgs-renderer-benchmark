#!/usr/bin/env python3
import json,sys
CELLS=("room@10000","room@15000","bicycle@10000","bicycle@15000")
QS=("0.05","0.10")
def ok_strong(x):
    return (x["loss_parity_pass"] and x["bias_rel"]<=.02 and
            x["cos_median"]>=.90 and x["cos_p10"]>=.80 and
            x["variance_reduction_vs_ht"]>=2.0 and
            x["progress_capture_median"]>=.85 and x["progress_capture_p10"]>=.70 and
            x["peer_delta_fraction_median"]<=.005 and
            x["truehold_psnr_delta_median_db"]>=-.02 and
            x["truehold_psnr_delta_worst_db"]>=-.08 and
            x["predicted_cost_fraction"]<=.25 and
            x["predicted_e2e_speedup"]>=3.0 and x["nan_inf_count"]==0)
def ok_promote(x):
    return (x["loss_parity_pass"] and x["bias_rel"]<=.03 and
            x["cos_median"]>=.85 and x["cos_p10"]>=.70 and
            x["variance_reduction_vs_ht"]>=1.5 and
            x["progress_capture_median"]>=.75 and x["progress_capture_p10"]>=.55 and
            x["truehold_psnr_delta_median_db"]>=-.04 and
            x["truehold_psnr_delta_worst_db"]>=-.12 and
            x["predicted_cost_fraction"]<=.30 and
            x["predicted_e2e_speedup"]>=3.0 and x["nan_inf_count"]==0)
d=json.load(open(sys.argv[1]))
if not d.get("loss_decomposition_global_pass",False):
    out={"verdict":"UCR0_LOSS_DECOMPOSITION_FAIL","selected_q":None}
else:
    strong=[]; promote=[]
    for q in QS:
        xs=[d["cells"][c]["q"][q] for c in CELLS]
        if all(ok_strong(x) for x in xs): strong.append(q)
        if all(ok_promote(x) for x in xs): promote.append(q)
    if strong: v="UCR0_STRONG_PROMOTE"; sel=strong[0]
    elif promote: v="UCR0_PROMOTE"; sel=promote[0]
    else: v="UCR0_DROP"; sel=None
    out={"verdict":v,"selected_q":sel,"strong_q":strong,"promote_q":promote}
open(sys.argv[1].replace("SUMMARY","VERDICT"),"w").write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
