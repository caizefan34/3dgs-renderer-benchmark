#!/usr/bin/env python3
import json, sys
from pathlib import Path

EXPECTED_CELLS = {"room@20000","room@25000","bicycle@20000","bicycle@25000"}
FRACTIONS = ("0.10","0.20")

def ok_strong(x, cache):
    return (
        x["utility_capture_median"] >= 0.80 and
        x["exact_repair_hit_fraction_median"] <= 0.25 and
        x["exact_repair_hit_fraction_p90"] <= 0.40 and
        x["R_loss_median"] >= 0.75 and
        x["R_loss_p10"] >= 0.50 and
        cache <= 16.0 and
        x["S_oracle_upper"] >= 3.0 and
        x["non_descent_count"] <= 2 and
        x.get("n_cameras", 8) == 8
    )

def ok_promote(x, cache):
    return (
        x["utility_capture_median"] >= 0.75 and
        x["exact_repair_hit_fraction_median"] <= 0.40 and
        x["exact_repair_hit_fraction_p90"] <= 0.55 and
        x["R_loss_median"] >= 0.65 and
        x["R_loss_p10"] >= 0.40 and
        cache <= 16.0 and
        x["S_oracle_upper"] >= 2.0 and
        x["non_descent_count"] <= 2 and
        x.get("n_cameras", 8) == 8
    )

def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: ir0_analyze.py IR0_SUMMARY.json")
    p = Path(sys.argv[1])
    d = json.loads(p.read_text())
    cells = d.get("cells", {})
    missing = sorted(EXPECTED_CELLS - set(cells))
    extra = sorted(set(cells) - EXPECTED_CELLS)
    if missing or extra:
        verdict = {
            "verdict":"IR0_INPUT_INVALID",
            "missing_cells":missing,
            "extra_cells":extra,
        }
    else:
        strong_by_f, promote_by_f, detail = {}, {}, {}
        for f in FRACTIONS:
            detail[f] = {}
            strong = True
            promote = True
            for c in sorted(EXPECTED_CELLS):
                cache = float(cells[c]["cache_GiB_32B_B32"])
                x = cells[c]["fractions"][f]
                s = ok_strong(x, cache)
                q = ok_promote(x, cache)
                detail[f][c] = {"strong":s, "promote":q}
                strong &= s
                promote &= q
            strong_by_f[f] = strong
            promote_by_f[f] = promote
        strong_fs = [f for f,v in strong_by_f.items() if v]
        promote_fs = [f for f,v in promote_by_f.items() if v]
        if strong_fs:
            v = "IR0_STRONG_PROMOTE"
            selected = strong_fs[0]
        elif promote_fs:
            v = "IR0_PROMOTE"
            selected = promote_fs[0]
        else:
            v = "IR0_DROP"
            selected = None
        verdict = {
            "verdict":v,
            "selected_fixed_fraction":selected,
            "strong_by_fraction":strong_by_f,
            "promote_by_fraction":promote_by_f,
            "detail":detail,
        }
    out = p.with_name("IR0_VERDICT.json")
    out.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))

if __name__ == "__main__":
    main()
