#!/usr/bin/env python3
import json,sys
from pathlib import Path
CELLS={"room@10000","room@15000","bicycle@10000","bicycle@15000"}
MS=("2","4")
def strong(x):
 return (x["valid_source_descent_count"]>=32 and x["E_span_median"]>=0.80
  and x["E_span_p10"]>=0.55 and x["P_src_median"]>=0.85
  and x["P_src_p10"]>=0.60 and x["peer_loss_win_rate"]>=0.45
  and x["truehold_psnr_delta_median_db"]>=-0.02
  and x["truehold_psnr_delta_p10_db"]>=-0.10
  and x["invalid_fraction"]<=0.05)
def promote(x):
 return (x["valid_source_descent_count"]>=32 and x["E_span_median"]>=0.65
  and x["E_span_p10"]>=0.40 and x["P_src_median"]>=0.70
  and x["P_src_p10"]>=0.40 and x["peer_loss_win_rate"]>=0.35
  and x["truehold_psnr_delta_median_db"]>=-0.05
  and x["truehold_psnr_delta_p10_db"]>=-0.15
  and x["invalid_fraction"]<=0.10)
def main():
 if len(sys.argv)!=2: raise SystemExit("usage: s0_analyze.py S0_SUMMARY.json")
 p=Path(sys.argv[1]); d=json.loads(p.read_text()); cells=d.get("cells",{})
 missing=sorted(CELLS-set(cells)); extra=sorted(set(cells)-CELLS)
 if missing or extra: out={"verdict":"S0_INPUT_INVALID","missing_cells":missing,"extra_cells":extra}
 else:
  detail={}; ss=[]; pp=[]
  for m in MS:
   detail[m]={}; sa=pa=True
   for c in sorted(CELLS):
    x=cells[c]["memory"][m]; s=strong(x); q=promote(x)
    detail[m][c]={"strong":s,"promote":q}; sa &= s; pa &= q
   if sa: ss.append(m)
   if pa: pp.append(m)
  if ss: verdict,sel="S0_STRONG_PROMOTE",ss[0]
  elif pp: verdict,sel="S0_PROMOTE",pp[0]
  else: verdict,sel="S0_DROP",None
  out={"verdict":verdict,"selected_fixed_memory":sel,"strong_memories":ss,"promote_memories":pp,"detail":detail}
 p.with_name("S0_VERDICT.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
