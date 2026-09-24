import ast, glob, os

cands = []
for pat in [
    "/mnt/storage_pool/3dgs-renderer-benchmark/repo/benchmark/run_*train_benchmark.py",
    "/mnt/storage_pool/3dgs-renderer-benchmark/**/run_*train_benchmark.py",
]:
    cands += glob.glob(pat, recursive=True)
cands = [c for c in cands if os.path.isfile(c)]
print("CANDS:", cands)
P = cands[0] if cands else None
if not P:
    print("NONE")
    raise SystemExit(1)
print("USING:", P)
src = open(P).read()
tree = ast.parse(src)
wanted = set(["_std_ll_forward", "make_forward_fn", "run_profile", "main"])
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
        print(f"===== {node.name} @L{node.lineno} =====")
        seg = src.splitlines()[node.lineno - 1 : node.end_lineno]
        for i, l in enumerate(seg, node.lineno):
            print(f"{i}: {l}")
        print()