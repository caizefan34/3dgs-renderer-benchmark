#!/usr/bin/env python3
import json,sys
from pathlib import Path
CELLS={"room@10000","room@15000","bicycle@10000","bicycle@15000"}
VARS=("FR-GLOBAL","FR-LOCAL")
def strong(x):
 return (x["n_probes"]==8 and x["correctness_pass"] and x["nan_inf_count"]==0
  and x["valid_source_descent_count"]>=7 and x["R_src_median"]>=1.30
  and x["R_src_p10"]>=1.00 and x["peer_win_rate"]>=0.65
  and x["truehold_win_rate"]>=0.60 and x["truehold_psnr_delta_median_db"]>=0
  and x["truehold_psnr_delta_worst_db"]>=-0.05)
def promote(x):
 return (x["n_probes"]==8 and x["correctness_pass"] and x["nan_inf_count"]==0
  and x["valid_source_descent_count"]>=6 and x["R_src_median"]>=1.15
  and x["R_src_p10"]>=0.80 and x["peer_win_rate"]>=0.55
  and x["truehold_win_rate"]>=0.55 and x["truehold_psnr_delta_median_db"]>=-0.02
  and x["truehold_psnr_delta_worst_db"]>=-0.10)
def main():
 if len(sys.argv)!=2: raise SystemExit("usage: ig0_analyze.py IG0_SUMMARY.json")
 p=Path(sys.argv[1]); d=json.loads(p.read_text()); cells=d.get("cells",{})
 missing=sorted(CELLS-set(cells)); extra=sorted(set(cells)-CELLS)
 if missing or extra: out={"verdict":"IG0_INPUT_INVALID","missing_cells":missing,"extra_cells":extra}
 else:
  detail={}; ss=[]; pp=[]
  for v in VARS:
   detail[v]={}; sa=pa=True
   for c in sorted(CELLS):
    x=cells[c]["variants"][v]; s=strong(x); q=promote(x)
    detail[v][c]={"strong":s,"promote":q}; sa &= s; pa &= q
   if sa: ss.append(v)
   if pa: pp.append(v)
  if ss: verdict,sel="IG0_STRONG_PROMOTE",ss[0]
  elif pp: verdict,sel="IG0_PROMOTE",pp[0]
  else: verdict,sel="IG0_DROP",None
  out={"verdict":verdict,"selected_fixed_variant":sel,"strong_variants":ss,"promote_variants":pp,"detail":detail}
 p.with_name("IG0_VERDICT.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
