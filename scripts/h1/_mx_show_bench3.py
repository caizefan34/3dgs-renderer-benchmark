import glob, os

roots = ["/mnt/storage_pool/3dgs-renderer-benchmark/repo/benchmark",
         "/mnt/storage_pool/3dgs-renderer-benchmark/repo/src"]
cands = []
for r in roots:
    if os.path.isdir(r):
        cands += glob.glob(r + "/run_*benchmark.py")
print("CANDIDATES:", cands)
for c in cands:
    txt = open(c, errors="replace").read()
    if "def _std_ll_forward" in txt:
        print("TARGET:", c)
        lns = txt.splitlines()
        for i, line in enumerate(lns):
            if "def _std_ll_forward" in line:
                for j in range(i, min(i + 75, len(lns))):
                    print(j + 1, ":", lns[j])
                break
        break
