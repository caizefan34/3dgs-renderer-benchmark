#!/usr/bin/env python3
import json,sys
from pathlib import Path
CELLS={"room@10000","room@15000","room@20000","bicycle@10000","bicycle@15000","bicycle@20000"}
def strong(x):
 return (x["valid_rho_count"]>=6 and x["rho_EC_median"]>=.70 and x["delta_rho_median"]>=.20
 and x["valid_auc_count"]>=4 and x["AUC_C_median"]>=.80 and x["delta_AUC_median"]>=.10
 and x["G_C_median"]>=1.30 and x["G_C_p10"]>=.90
 and x["truehold_psnr_delta_median_db"]>=-.02 and x["invalid_fraction"]<=.05)
def promote(x):
 return (x["valid_rho_count"]>=6 and x["rho_EC_median"]>=.55 and x["delta_rho_median"]>=.10
 and x["valid_auc_count"]>=4 and x["AUC_C_median"]>=.70 and x["delta_AUC_median"]>=.05
 and x["G_C_median"]>=1.15 and x["G_C_p10"]>=.75
 and x["truehold_psnr_delta_median_db"]>=-.05 and x["invalid_fraction"]<=.10)
def main():
 if len(sys.argv)!=2: raise SystemExit("usage: et0_analyze.py ET0_SUMMARY.json")
 p=Path(sys.argv[1]); d=json.loads(p.read_text()); cells=d.get("cells",{})
 miss=sorted(CELLS-set(cells)); extra=sorted(set(cells)-CELLS)
 if miss or extra: out={"verdict":"ET0_INPUT_INVALID","missing_cells":miss,"extra_cells":extra}
 else:
  det={}; sa=pa=True
  for c in sorted(CELLS):
   s=strong(cells[c]); q=promote(cells[c]); det[c]={"strong":s,"promote":q}; sa&=s; pa&=q
  out={"verdict":"ET0_STRONG_PROMOTE" if sa else ("ET0_PROMOTE" if pa else "ET0_DROP"),"detail":det}
 p.with_name("ET0_VERDICT.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
