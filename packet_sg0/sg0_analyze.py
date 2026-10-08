#!/usr/bin/env python3
import json, sys
from pathlib import Path

CELLS = {"room@20000","room@25000","bicycle@20000","bicycle@25000"}
CONFIGS = ("L32/SIM13","L32/AFF22","L128/SIM13","L128/AFF22")

def strong(x):
    k4, k8 = x["K4"], x["K8"]
    return (
        k4["p_hold_median"] >= 0.85 and
        k4["p_hold_p10"] >= 0.60 and
        k8["p_hold_median"] >= 0.80 and
        k8["p_hold_p10"] >= 0.55 and
        k8["p_design_median"] >= 0.80 and
        x["coarse_dof_ratio_median"] <= 0.05 and
        k8["oracle_improve_count"] >= 7 and
        k8["base_non_descent_count"] <= 2 and
        k4.get("n_rollouts",8) == 8 and
        k8.get("n_rollouts",8) == 8
    )

def promote(x):
    k4, k8 = x["K4"], x["K8"]
    return (
        k4["p_hold_median"] >= 0.75 and
        k4["p_hold_p10"] >= 0.40 and
        k8["p_hold_median"] >= 0.65 and
        k8["p_hold_p10"] >= 0.30 and
        k8["p_design_median"] >= 0.70 and
        x["coarse_dof_ratio_median"] <= 0.10 and
        k8["oracle_improve_count"] >= 6 and
        k8["base_non_descent_count"] <= 2 and
        k4.get("n_rollouts",8) == 8 and
        k8.get("n_rollouts",8) == 8
    )

def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: sg0_analyze.py SG0_SUMMARY.json")
    p = Path(sys.argv[1])
    d = json.loads(p.read_text())
    cells = d.get("cells", {})
    missing = sorted(CELLS - set(cells))
    extra = sorted(set(cells) - CELLS)
    if missing or extra:
        out = {"verdict":"SG0_INPUT_INVALID","missing_cells":missing,"extra_cells":extra}
    else:
        details = {}
        strong_configs = []
        promote_configs = []
        for cfg in CONFIGS:
            details[cfg] = {}
            s_all = True
            p_all = True
            for cell in sorted(CELLS):
                x = cells[cell]["configs"][cfg]
                s = strong(x)
                q = promote(x)
                details[cfg][cell] = {"strong":s,"promote":q}
                s_all &= s
                p_all &= q
            if s_all:
                strong_configs.append(cfg)
            if p_all:
                promote_configs.append(cfg)
        if strong_configs:
            verdict = "SG0_STRONG_PROMOTE"
            selected = strong_configs[0]
        elif promote_configs:
            verdict = "SG0_PROMOTE"
            selected = promote_configs[0]
        else:
            verdict = "SG0_DROP"
            selected = None
        out = {
            "verdict": verdict,
            "selected_fixed_config": selected,
            "strong_configs": strong_configs,
            "promote_configs": promote_configs,
            "details": details,
        }
    out_path = p.with_name("SG0_VERDICT.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
