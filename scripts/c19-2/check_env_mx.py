import sys, os, subprocess

print("Running from:", sys.executable)
print("Python version:", sys.version)

# Try to import torch
try:
    import torch
    print("torch OK:", torch.__version__, "cuda:", torch.version.cuda)
    import gsplat
    print("gsplat OK:", gsplat.__version__)
except ImportError as e:
    print("Import failed:", e)

# Search for other python3 binaries with torch
print("\nSearching for python3 with torch...")
search_dirs = [
    "/home/liaoyuanjun/.local/bin",
    "/home/liaoyuanjun/miniconda3/bin",
    "/home/liaoyuanjun/miniconda/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/opt/conda/bin",
]

for d in search_dirs:
    python_path = os.path.join(d, "python3")
    if os.path.exists(python_path):
        r = subprocess.run(
            [python_path, "-c", "import torch, gsplat; print(torch.__file__, gsplat.__version__)"],
            capture_output=True, text=True, timeout=5
        )
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  {python_path}: {status} -> {r.stdout.strip()[:80] if r.stdout else r.stderr[:80]}")

print("\nChecking pip packages...")
r = subprocess.run(
    [sys.executable, "-m", "pip", "list", "--format=columns"],
    capture_output=True, text=True, timeout=10
)
for line in r.stdout.split("\n"):
    if "torch" in line.lower() or "gsplat" in line.lower():
        print(" ", line)

print("\nDone")
