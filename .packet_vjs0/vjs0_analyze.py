#!/usr/bin/env python3
import json,sys
CELLS=("room@10000","room@15000","bicycle@10000","bicycle@15000")
def strong(x):
    return (x["valid_progress_count"]>=7 and x["progress_capture_median"]>=.90 and
            x["progress_capture_p10"]>=.75 and x["truehold_psnr_delta_median_db"]>=-.02 and
            x["truehold_psnr_delta_worst_db"]>=-.08 and x["rgb_linearization_error_median"]<=.03 and
            x["predicted_speedup"]>=3.0 and x["inner_cost_fraction"]<=.15 and x["nan_inf_count"]==0)
def promote(x):
    return (x["valid_progress_count"]>=6 and x["progress_capture_median"]>=.80 and
            x["progress_capture_p10"]>=.60 and x["truehold_psnr_delta_median_db"]>=-.04 and
            x["truehold_psnr_delta_worst_db"]>=-.12 and x["rgb_linearization_error_median"]<=.05 and
            x["predicted_speedup"]>=2.5 and x["inner_cost_fraction"]<=.20 and x["nan_inf_count"]==0)
d=json.load(open(sys.argv[1]))
if not d.get("linearization_parity_pass",False):
    out={"verdict":"VJS0_LINEARIZATION_IMPLEMENTATION_FAIL"}
else:
    xs=[d["cells"][c]["K4"] for c in CELLS]
    s=all(strong(x) for x in xs); p=all(promote(x) for x in xs)
    out={"verdict":"VJS0_STRONG_PROMOTE" if s else ("VJS0_PROMOTE" if p else "VJS0_DROP"),
         "k4_strong":s,"k4_promote":p}
open(sys.argv[1].replace("SUMMARY","VERDICT"),"w").write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
