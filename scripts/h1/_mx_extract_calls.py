"""Extract exact call conventions for gsplat rasterization from the benchmark script."""
import ast, textwrap

SRC = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/benchmark/run_higs_train_benchmark.py"

with open(SRC) as fh:
    src = fh.read()

# Find the call sites for rasterization-related functions and print context around them
import re

patterns = [
    r"fully_projected?\(",
    r"fully_fusion_projected\(",
    r"rasterize_gaussian_higs",
    r"rasterization\(",
    r"create_higs_renderer\(",
    r"backward_mode=",
]

for pat in patterns:
    for m in re.finditer(pat, src):
        start = max(0, m.start() - 600)
        end = min(len(src), m.end() + 600)
        snippet = src[start:end]
        # Get line numbers
        lineno = src[:m.start()].count("\n") + 1
        print(f"\n--- pattern '{pat}' at line {lineno} ---")
        print(snippet)
        print("...")
