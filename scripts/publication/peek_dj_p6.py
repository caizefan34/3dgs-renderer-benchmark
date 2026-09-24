import json
PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"
F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"

def res(p):
    d = json.load(open(p + "/results.json"))
    return d["final_eval"]["psnr"], d["timing"]["total_wall_s"]

rows = {}
for seed, b, c in [
    (42, F30K + "/b1a_drjohnson", F30K + "/c0_drjohnson"),
    (43, PUB + "/b1as43_drjohnson", PUB + "/c0s43_drjohnson"),
    (44, PUB + "/b1as44_drjohnson", PUB + "/c0s44_drjohnson"),
]:
    bp, bw = res(b)
    cp, cw = res(c)
    rows[seed] = (bp, cp, cp - bp, bw / cw)
    print("s%d drjohnson: b1a=%.3f c0=%.3f dPSNR=%+.3f speedup=%.4f" % (seed, bp, cp, cp - bp, bw / cw))

deltas = [rows[s][2] for s in (42, 43, 44)]
print("\nR-13 drjohnson deltas: %s" % " ".join("%+.3f" % d for d in deltas))
print("all > +0.5 dB:", all(d > 0.5 for d in deltas))
print("all > 0:", all(d > 0 for d in deltas))
