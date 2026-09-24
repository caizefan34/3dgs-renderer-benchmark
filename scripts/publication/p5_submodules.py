import json
import subprocess

BASE = "/mnt/storage_pool/liaoyuanjun/strong_baselines"
OUT = "/mnt/storage_pool/liaoyuanjun/pubphase/aggregates/p5_external.json"

def git(repo, *args):
    return subprocess.check_output(["git", "-C", repo, *args], text=True).strip()

prov = {}
for b in ("faster-gs", "fastgs", "speedy-splat"):
    repo = f"{BASE}/{b}"
    entry = {"commit": git(repo, "rev-parse", "HEAD"),
             "commit_subject": git(repo, "log", "-1", "--format=%s")}
    subs = git(repo, "submodule", "status").splitlines()
    submap = {}
    for line in subs:
        line = line.strip()
        if line:
            sha = line.split()[0]
            name = line.split()[1] if len(line.split()) > 1 else "?"
            submap[name] = sha
    if submap:
        entry["submodules"] = submap
    # untracked-only dirtiness is ignorable; record tracked modifications
    st = git(repo, "status", "--porcelain").splitlines()
    tracked = [l for l in st if not l.startswith("??")]
    entry["tracked_modifications"] = tracked
    prov[b] = entry
    print(b, json.dumps(entry))

d = json.load(open(OUT))
d["provenance_detail"] = prov
with open(OUT, "w") as f:
    json.dump(d, f, indent=1)
print("UPDATED", OUT)
