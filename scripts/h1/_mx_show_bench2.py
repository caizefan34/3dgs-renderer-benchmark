import re

BENCH = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/benchmark/run_himsg_train_benchmark.py"
lns = open(BENCH).read().splitlines()

def dump(a, b, label=""):
    a = max(0, a)
    b = min(len(lns), b)
    print("=" * 60)
    print(f"--- {label}: lines {a+1}-{b} ---")
    for i in range(a, b):
        print(f"{i+1}: {lns[i]}")

for i, line in enumerate(lns):
    if line.startswith("def ") and any(k in line for k in ["main", "eval", "profile", "run"]):
        print(f"def at {i+1}: {line.strip()}")

print("\n# viewmats occurrences:")
for i, line in enumerate(lns):
    if "viewmats" in line and ("=" in line or "[" in line):
        print(f"{i+1}: {line.strip()}")

for i, line in enumerate(lns):
    if "def _std_ll_forward" in line:
        dump(i, i + 55, "_std_ll_forward")
        break
