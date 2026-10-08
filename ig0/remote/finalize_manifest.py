import json, os, hashlib
OUT = "/root/ig0_run/out"
ROOT = "/root/ig0_run"
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
entries = []
for nm in sorted(os.listdir(OUT)):
    if nm == "MANIFEST.sha256":
        continue
    entries.append((sha256_file(os.path.join(OUT, nm)), "out/" + nm))
for nm in ["ig0_fisher.py", "ig0_reference_checks.py", "ig0_fisher_torch.py",
           "ig0_gpu_probe.py", "ig0_analyze.py", "full_grid.log"]:
    p = os.path.join(ROOT, nm)
    if os.path.exists(p):
        entries.append((sha256_file(p), nm))
lines = ["%s  %s" % (h, rel) for h, rel in entries]
lines.append("# self-entry excluded: out/MANIFEST.sha256")
with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("manifest entries:", len(entries))
