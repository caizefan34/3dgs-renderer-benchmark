"""Pretty-print one P-TRANSITION unit JSON."""
import json
import sys

path = sys.argv[1]
d = json.load(open(path))
print("scene", d["scene"], "iter", d["iteration"], "seed", d["seed"], "status", d["status"])
print("camera_ids", d["camera_ids"])
print("selection", {k: v for k, v in d["selection"].items() if k != "ids_sha256"})
print("before_per_view_loss", [round(x, 6) for x in d["before_per_view_loss"]])
print("position_observability", {k: round(v, 9) if isinstance(v, float) else v
                                 for k, v in d["position_observability"].items()})
print("shared_setup_seconds", round(d["shared_setup_seconds"], 3))
print("exec_order:", [(c["chart"], c["lr"]) for c in d["execution_order"]])
for c in d["candidates"]:
    print("  %-8s lr=%-6s train_delta=%+.6f audit_delta=%+.6f eta=%.6f opt_s=%.2f with_eval_s=%.2f peak_MB=%.0f"
          % (c["chart"], c["lr"], c["train_delta"], c["audit_delta"], c["eta"],
             c["optimization_seconds"], c["with_evaluation_seconds"],
             c["peak_allocated_bytes"] / 2**20))
    print("      losses", [round(x, 6) for x in c["losses"]],
          "fit_history_len", len(c["fit_history"]), "trace_len", len(c["trace"]))
print("best_by_eta", json.dumps(d["best_by_eta"], indent=1))
print("R_eta", d["R_eta"], "screen_memory_ratio", d["screen_memory_ratio"])
