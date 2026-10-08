#!/usr/bin/env python3
"""Aggregate 4 per-cell CG0_SUMMARY.json into the analyzer contract + verdict."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg0_analyze import main as analyze_main, CELLS

def main():
    in_dir = sys.argv[1]
    cells = {}
    top = {"source_identity_pass": True, "no_oracle_leakage_pass": True,
           "oracle_extraction_pass": True, "compiler_numerical_pass": True}
    for c in CELLS:
        p = os.path.join(in_dir, c, "CG0_SUMMARY.json")
        with open(p) as f:
            cells[c] = json.load(f)
    top["cells"] = cells
    out = os.path.join(in_dir, "CG0_SUMMARY.json")
    with open(out, "w") as f:
        json.dump(top, f, indent=2)
    print(f"[cg0] wrote {out}")
    # run frozen analyzer
    sys.argv = ["cg0_analyze.py", out]
    analyze_main()

if __name__ == "__main__":
    main()
