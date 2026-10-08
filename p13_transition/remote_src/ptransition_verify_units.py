"""Verify all 12 P-TRANSITION unit JSONs: status, inputs, GPU records."""
import glob
import json

for f in sorted(glob.glob("/root/epic05_ptransition/results/transition_*.json")):
    d = json.load(open(f))
    gb = [l for l in d.get("gpu_before", []) if l.startswith("3,")]
    ga = [l for l in d.get("gpu_after", []) if l.startswith("3,")]
    print(f.split("/")[-1])
    print("  status:", d["status"], "| cameras:", d["camera_ids"],
          "| fused:", d["fused_loss"]["so_sha256"][:12],
          "| exec_order_n:", len(d["execution_order"]))
    print("  input_sha:", {k.split("/")[-1]: v[:12] for k, v in d["input_sha256"].items()})
    print("  gpu3 before:", gb[0] if gb else None)
    print("  gpu3 after :", ga[0] if ga else None)
