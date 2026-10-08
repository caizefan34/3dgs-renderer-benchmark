"""Compact per-unit table for the P-TRANSITION return package.

For each unit: selection, per-candidate chart/lr/train_delta/eta/opt_s/with_eval_s,
the deterministically shuffled execution order (first-executed marked), and the
per-chart best. Also quantifies whether the first-executed candidate paid a
one-time warm-up in optimization time.
"""
import glob
import json

rows = []
for f in sorted(glob.glob("/root/epic05_ptransition/results/transition_*.json")):
    d = json.load(open(f))
    if d.get("status") != "P_TRANSITION_UNIT_COMPLETE":
        continue
    first = (d["execution_order"][0]["chart"], d["execution_order"][0]["lr"])
    cands = []
    for c in d["candidates"]:
        cands.append({
            "chart": c["chart"], "lr": c["lr"],
            "first_executed": c["chart"] == first[0] and c["lr"] == first[1],
            "train_delta": c["train_delta"], "audit_delta": c["audit_delta"],
            "train_gain": c["train_gain"], "eta": c["eta"],
            "audit_per_train": c["audit_per_train"],
            "opt_s": c["optimization_seconds"], "with_eval_s": c["with_evaluation_seconds"],
            "all_in_s": c["all_in_seconds"],
            "peak_MB": c["peak_allocated_bytes"] / 2 ** 20,
        })
    rows.append({
        "file": f.split("/")[-1], "scene": d["scene"], "iteration": d["iteration"],
        "seed": d["seed"], "camera_ids": d["camera_ids"],
        "N": d["selection"]["N"], "common_visible": d["selection"]["common_visible"],
        "active": d["selection"]["active"],
        "shared_setup_s": d["shared_setup_seconds"],
        "first_executed": {"chart": first[0], "lr": first[1]},
        "candidates": cands,
        "best_by_eta": d["best_by_eta"], "R_eta": d["R_eta"],
        "screen_memory_ratio": d["screen_memory_ratio"],
    })

out = {"n_units": len(rows), "units": rows}
dest = "/root/epic05_ptransition/PTRANSITION_UNIT_TABLE.json"
open(dest, "w").write(json.dumps(out, indent=2))

for r in rows:
    print("%-32s N=%d active=%d" % (r["file"], r["N"], r["active"]))
    print("   first executed: %s lr=%s | R_eta %.4f | mem_ratio %.4f"
          % (r["first_executed"]["chart"], r["first_executed"]["lr"], r["R_eta"], r["screen_memory_ratio"]))
    for c in r["candidates"]:
        print("   %-8s lr=%-6s gain=%+.3e eta=%.3e opt_s=%.2f w_eval=%.2f%s"
              % (c["chart"], c["lr"], c["train_gain"], c["eta"], c["opt_s"], c["with_eval_s"],
                 " <-FIRST" if c["first_executed"] else ""))
    b = r["best_by_eta"]
    print("   best: screen lr=%s eta=%.3e | geometry lr=%s eta=%.3e"
          % (b["screen"]["lr"], b["screen"]["eta"], b["geometry"]["lr"], b["geometry"]["eta"]))
print("wrote", dest)
