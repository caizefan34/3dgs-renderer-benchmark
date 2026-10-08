#!/usr/bin/env python3
import json, sys
from pathlib import Path

CELLS={
 "room@10000","room@15000","room@20000",
 "bicycle@10000","bicycle@15000","bicycle@20000"
}

def strong(x):
    return (
      x["n_panels"]==3 and x["complete"] and x["nan_inf_count"]==0 and
      x["T_oracle_median"]>0 and
      x["G_random_median"]>=0.60 and
      x["G_hard_median"]>=0.35 and
      x["G_true_median"]>=0.40 and
      x["H_oracle_positive_count"]==3 and
      x["F_pos_median"]>=0.75 and
      x["rho_TH_median"]>=0.40
    )

def promote(x):
    return (
      x["n_panels"]==3 and x["complete"] and x["nan_inf_count"]==0 and
      x["T_oracle_median"]>0 and
      x["G_random_median"]>=0.35 and
      x["G_hard_median"]>=0.20 and
      x["G_true_median"]>=0.20 and
      x["H_oracle_positive_count"]>=2 and
      x["F_pos_median"]>=0.65 and
      x["rho_TH_median"]>=0.20
    )

def main():
    if len(sys.argv)!=2: raise SystemExit("usage: cvt0_analyze.py CVT0_SUMMARY.json")
    p=Path(sys.argv[1]); d=json.loads(p.read_text())
    cells=d.get("cells",{})
    missing=sorted(CELLS-set(cells)); extra=sorted(set(cells)-CELLS)
    if missing or extra:
        out={"verdict":"CVT0_INPUT_INVALID","missing_cells":missing,"extra_cells":extra}
    else:
        detail={}; sa=True; pa=True
        for c in sorted(CELLS):
            s=strong(cells[c]); q=promote(cells[c])
            detail[c]={"strong":s,"promote":q}; sa &= s; pa &= q
        out={"verdict":"CVT0_STRONG_PROMOTE" if sa else ("CVT0_PROMOTE" if pa else "CVT0_DROP"),
             "detail":detail}
    p.with_name("CVT0_VERDICT.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
