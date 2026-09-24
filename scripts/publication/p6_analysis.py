#!/usr/bin/env python3
"""P6 multi-seed analysis: executes the pre-registered 4-step framework when all 16
P6 runs are present. Idempotent; exits 1 with a missing-list if incomplete.

P6 cohort: {drjohnson, train, bicycle, room} x {b1a, c0} x {s43, s44}.
Seed-42 references come from FINAL-30K (b1a_*, c0_*).
"""
import json
import math
import sys

PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"
F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
OUT = f"{PUB}/p6_results.json"

P6_SCENES = ["drjohnson", "train", "bicycle", "room"]
SEEDS = {"42": ("{f}/b1a_{s}", "{f}/c0_{s}"),
         "43": ("{p}/b1as43_{s}", "{p}/c0s43_{s}"),
         "44": ("{p}/b1as44_{s}", "{p}/c0s44_{s}")}

def res(path):
    d = json.load(open(f"{path}/results.json"))
    return {"psnr": d["final_eval"]["psnr"],
            "ssim": d["final_eval"]["ssim"],
            "lpips": d["final_eval"].get("lpips"),
            "wall": d["timing"]["total_wall_s"],
            "n": d["final_eval"]["n_gaussians"]}


def contamination_flags():
    """Map run-dir name -> contaminated flag from the scheduler state."""
    try:
        st = json.load(open("/mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler_state.json"))
        return {d["rid"]: bool(d.get("contaminated")) for d in st.get("done", [])}
    except FileNotFoundError:
        return {}


CONTAM = contamination_flags()

missing = []
data = {}
for s in P6_SCENES:
    data[s] = {}
    for k, (bpat, cpat) in SEEDS.items():
        bp = (bpat.format(f=F30K, p=PUB, s=s))
        cp = (cpat.format(f=F30K, p=PUB, s=s))
        for p in (bp, cp):
            try:
                res(p)
            except FileNotFoundError:
                missing.append(p)

# contamination guard: any P6 run carrying the sticky flag is a protocol failure
# for timing; refuse to emit numbers that include it.
flagged = []
for s in P6_SCENES:
    for k, (bpat, cpat) in SEEDS.items():
        for pat in (bpat, cpat):
            rid = pat.format(f="", p="", s=s).strip("/").split("/")[-1]
            if CONTAM.get(rid):
                flagged.append(f"{rid} (seed {k}, {s})")
if flagged:
    print("CONTAMINATED P6 RUNS PRESENT - timing invalid, re-run required:")
    for f in flagged:
        print("  ", f)
    sys.exit(2)

if missing:
    print(f"P6 INCOMPLETE ({len(missing)} missing):")
    for m in missing:
        print(" ", m)
    sys.exit(1)

print("P6 COMPLETE: 4 scenes x 2 arms x 3 seeds (16 new + 8 FINAL-30K refs)\n")

# 1. Speedup stability
print("== 1. Speedup stability (wall_b1a / wall_c0, S19) ==")
geos = {}
for k in ("42", "43", "44"):
    ratios = []
    for s in P6_SCENES:
        b, c = data[s].get(k), data[s].get(k)
        b = res(SEEDS[k][0].format(f=F30K, p=PUB, s=s))
        c = res(SEEDS[k][1].format(f=F30K, p=PUB, s=s))
        data[s][k] = (b, c)
        ratios.append(b["wall"] / c["wall"])
        print(f"  s{k} {s:11s} b1a={b['wall']:8.1f} c0={c['wall']:8.1f} speedup={b['wall']/c['wall']:.4f}")
    g = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    geos[k] = g
    print(f"  s{k} 4-scene geomean: {g:.4f}  (all>1: {all(r > 1 for r in ratios)})")
f30k_band = "FINAL-30K per-scene speedups all >1; headline geomean 1.0685"
print(f"  reference: {f30k_band}")

# 2. Quality-delta stability
print("\n== 2. Quality-delta stability (psnr_c0 - psnr_b1a) ==")
for s in P6_SCENES:
    ds = []
    for k in ("42", "43", "44"):
        b, c = data[s][k]
        ds.append(c["psnr"] - b["psnr"])
    spread = max(ds) - min(ds)
    print(f"  {s:11s} dPSNR per seed: " + " ".join(f"{d:+.3f}" for d in ds) +
          f"  spread={spread:.3f}")

# 3. drjohnson seed test (R-13)
print("\n== 3. drjohnson R-13 seed test ==")
dj = []
for k in ("42", "43", "44"):
    b, c = data["drjohnson"][k]
    dj.append(c["psnr"] - b["psnr"])
    print(f"  s{k}: b1a={b['psnr']:.3f} c0={c['psnr']:.3f} delta={c['psnr']-b['psnr']:+.3f}")
persist = all(d > 0.5 for d in dj)
print(f"  persistence: {'REPRODUCIBLE (all seeds > +0.5 dB)' if persist else 'NOT seed-stable at the +0.5 threshold'}")

# 4. Aggregate-without-drjohnson sensitivity (4-scene subset)
print("\n== 4. 4-scene geomean without drjohnson ==")
for k in ("42", "43", "44"):
    ratios = [data[s][k][0]["wall"] / data[s][k][1]["wall"] for s in P6_SCENES if s != "drjohnson"]
    g = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    dq = [data[s][k][1]["psnr"] - data[s][k][0]["psnr"] for s in P6_SCENES if s != "drjohnson"]
    print(f"  s{k}: geomean={g:.4f} mean_dPSNR={sum(dq)/len(dq):+.3f}")

out = {"claim": "P6 multi-seed confirmation",
       "per_seed_geomeans": geos,
       "drjohnson_deltas": {"42": dj[0], "43": dj[1], "44": dj[2]},
       "drjohnson_persistent": persist,
       "per_scene": {s: {k: {"speedup": data[s][k][0]["wall"] / data[s][k][1]["wall"],
                             "d_psnr": data[s][k][1]["psnr"] - data[s][k][0]["psnr"]}
                        for k in ("42", "43", "44")} for s in P6_SCENES}}
json.dump(out, open(OUT, "w"), indent=1)
print(f"\nWROTE {OUT}")
