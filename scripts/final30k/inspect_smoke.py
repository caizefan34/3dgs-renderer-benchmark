#!/usr/bin/env python3
"""Quick inspection of a smoke results.json."""
import json
import sys

r = json.load(open(sys.argv[1]))
print("per_iter rows:", len(r["per_iter"]), "last:", r["per_iter"][-1])
print("events:", len(r["densification_events"]))
e = r["densification_events"][0]
print("event0:", {k: e[k] for k in ("iter", "n_before", "n_after", "cloned", "split",
                                    "pruned_total", "selected", "grad_nan_count", "grad_inf_count")})
print("event0 grad stats:", e["grad_norm_stats"])
print("event0 sel stats:", e["selected_grad_norm_stats"])
print("evals:", [(x["step"], round(x["psnr"], 2)) for x in r["eval_rows"]])
print("totals: clones", r["total_clones"], "splits", r["total_splits"], "prunes", r["total_prunes"])
print("final_eval:", r["final_eval"])
print("identity:", r["binary_identity"].get("so_sha256"))
print("renderer:", json.dumps(r["renderer"])[:200])
