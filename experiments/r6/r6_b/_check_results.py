import json, glob, os

print("=== R6-B correctness results on mx ===")
for f in sorted(glob.glob("/mnt/storage_pool/liaoyuanjun/r6_profiling/r6_b_correctness_*.json")):
    d = json.load(open(f))
    print(f"\n--- {os.path.basename(f)} ---")
    print("scene:", d.get("scene"))
    print("overall_pass:", d.get("overall_pass"))
    stale = d.get("stale_gradient_test", {})
    if isinstance(stale, dict):
        for k, v in stale.items():
            if k != "details":
                print(f"  {k}: {v}")

print("\n=== R6-B benchmark results ===")
for f in sorted(glob.glob("/tmp/r6b_bench*/**/*.json", recursive=True) + glob.glob("/tmp/**/r6_b_benchmark*.json", recursive=True)):
    print(f"\n--- {f} ---")
    try:
        d = json.load(open(f))
        print(json.dumps(d, indent=2)[:3000])
    except Exception as e:
        print(f"  ERR: {e}")

print("\n=== R6-B logs ===")
for f in sorted(glob.glob("/mnt/storage_pool/liaoyuanjun/r6_profiling/r6_b_*.log")):
    print(f"\n--- {os.path.basename(f)} ---")
    try:
        with open(f) as fh:
            lines = fh.readlines()
        print("".join(lines[-15:]))
    except Exception as e:
        print(f"  ERR: {e}")
