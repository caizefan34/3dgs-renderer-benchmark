#!/usr/bin/env python3
import json, sys
from pathlib import Path

CELLS = {"room@5000","room@15000","bicycle@5000","bicycle@15000"}

def strong(x):
    return (
        x["n_rollouts"] == 8 and
        x["atom_positive_count"] >= 7 and
        x["parent_positive_count"] >= 4 and
        x["r_parent_median"] is not None and
        x["r_parent_p10"] is not None and
        x["r_parent_median"] >= 2.0 and
        x["r_parent_p10"] >= 1.0 and
        x["p_full8_median"] >= 0.20 and
        x["atom_psnr_delta_median_db"] > 0.0 and
        x["atom_psnr_delta_worst_db"] >= -0.05 and
        x["candidate_visibility_median"] >= 4.0 and
        x["failure_count"] <= 1
    )

def promote(x):
    return (
        x["n_rollouts"] == 8 and
        x["atom_positive_count"] >= 6 and
        x["r_parent_median"] is not None and
        x["r_parent_p10"] is not None and
        x["r_parent_median"] >= 1.50 and
        x["r_parent_p10"] >= 0.75 and
        x["p_full8_median"] >= 0.10 and
        x["atom_psnr_delta_median_db"] >= -0.01 and
        x["atom_psnr_delta_worst_db"] >= -0.10 and
        x["candidate_visibility_median"] >= 3.0 and
        x["failure_count"] <= 2
    )

def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: ab0_analyze.py AB0_SUMMARY.json")
    p = Path(sys.argv[1])
    d = json.loads(p.read_text())
    cells = d.get("cells", {})
    missing = sorted(CELLS - set(cells))
    extra = sorted(set(cells) - CELLS)

    if missing or extra:
        out = {"verdict":"AB0_INPUT_INVALID","missing_cells":missing,"extra_cells":extra}
    else:
        detail = {}
        s_all = True
        p_all = True
        for c in sorted(CELLS):
            x = cells[c]
            s = strong(x)
            q = promote(x)
            detail[c] = {"strong":s,"promote":q}
            s_all &= s
            p_all &= q

        if s_all:
            verdict = "AB0_STRONG_PROMOTE"
        elif p_all:
            verdict = "AB0_PROMOTE"
        else:
            verdict = "AB0_DROP"

        out = {"verdict":verdict,"detail":detail}

    op = p.with_name("AB0_VERDICT.json")
    op.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
