#!/usr/bin/env python3
import json, sys
from pathlib import Path

CELLS = {
    "room@5000","room@15000","room@25000",
    "bicycle@5000","bicycle@15000","bicycle@25000"
}
FAMILIES = ("E2D","OPA")

def strong(x):
    k4, k8 = x["K4"], x["K8"]
    return (
        bool(x["parity_pass"]) and
        k4["p_hold_median"] >= 1.20 and
        k4["p_hold_p10"] >= 0.85 and
        k4["e_hold_median"] >= 1.15 and
        k8["p_hold_median"] >= 1.30 and
        k8["p_hold_p10"] >= 1.00 and
        k8["e_hold_median"] >= 1.20 and
        k8["p_design_median"] >= 1.20 and
        k8["psnr_delta_median_db"] >= -0.03 and
        k8["psnr_delta_worst_db"] >= -0.10 and
        k8["base_non_descent_count"] <= 2 and
        k8["nan_inf_count"] == 0 and
        k4.get("n_rollouts",8) == 8 and k8.get("n_rollouts",8) == 8
    )

def promote(x):
    k4, k8 = x["K4"], x["K8"]
    return (
        bool(x["parity_pass"]) and
        k4["p_hold_median"] >= 1.08 and
        k4["p_hold_p10"] >= 0.70 and
        k4["e_hold_median"] >= 1.03 and
        k8["p_hold_median"] >= 1.15 and
        k8["p_hold_p10"] >= 0.80 and
        k8["e_hold_median"] >= 1.08 and
        k8["p_design_median"] >= 1.10 and
        k8["psnr_delta_median_db"] >= -0.05 and
        k8["psnr_delta_worst_db"] >= -0.15 and
        k8["base_non_descent_count"] <= 2 and
        k8["nan_inf_count"] == 0 and
        k4.get("n_rollouts",8) == 8 and k8.get("n_rollouts",8) == 8
    )

def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: h0_analyze.py H0_SUMMARY.json")
    p = Path(sys.argv[1])
    d = json.loads(p.read_text())
    cells = d.get("cells", {})
    missing = sorted(CELLS - set(cells))
    extra = sorted(set(cells) - CELLS)
    if missing or extra:
        out = {"verdict":"H0_INPUT_INVALID","missing_cells":missing,"extra_cells":extra}
    else:
        detail = {}
        strong_f, promote_f = [], []
        for fam in FAMILIES:
            detail[fam] = {}
            s_all = True
            p_all = True
            for cell in sorted(CELLS):
                x = cells[cell]["families"][fam]
                s = strong(x)
                q = promote(x)
                detail[fam][cell] = {"strong":s,"promote":q}
                s_all &= s
                p_all &= q
            if s_all: strong_f.append(fam)
            if p_all: promote_f.append(fam)
        if strong_f:
            verdict, selected = "H0_STRONG_PROMOTE", strong_f[0]
        elif promote_f:
            verdict, selected = "H0_PROMOTE", promote_f[0]
        else:
            verdict, selected = "H0_DROP", None
        out = {
            "verdict":verdict,
            "selected_fixed_family":selected,
            "strong_families":strong_f,
            "promote_families":promote_f,
            "detail":detail
        }
    outp = p.with_name("H0_VERDICT.json")
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
