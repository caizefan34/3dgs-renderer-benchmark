#!/usr/bin/env python3
import json, sys
d = json.load(open(sys.argv[1]))
print("passed:", d.get("passed"))
if "rasterizer_sequential" in d:
    r = d["rasterizer_sequential"]
    print("stale_row_zero:", all(v["is_zero"] for v in r["stale_row_zero_b1"].values()))
    print("comparison_t1_b1:", {k: v["b1"]["max_abs"] for k, v in r["comparison_t1"].items()})
    print("comparison_t_b1:", {k: v["b1"]["max_abs"] for k, v in r["comparison_t"].items()})
