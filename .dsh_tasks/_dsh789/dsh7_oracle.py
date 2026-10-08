"""DSH-7 BARS0 structural oracle bound from Phase-A per-tile data.

Uses per-tile splat counts (instance_ranges end-start) saved in the raw npz to
compute exact bucket counts sum_t ceil(S_t / K) for K in {8,16,32,64}, the
per-tile adaptive oracle, and the short-tail occupancy. This is a STRUCTURAL
upper-bound on the K-dependent cost (checkpoint stores + backward buckets both
scale with bucket count); it does not measure wall time. If even this bound
cannot clear the B1 8% renderer threshold, Phase B timing cannot promote.
"""
import json
from pathlib import Path
import numpy as np

RUNS = Path("/root/dsh_20261001/runs/DSH_PHASEA")
SCENES = ["room", "garden", "bicycle"]
KS = [8, 16, 32, 64]
MATURE = [10000, 14500, 20000, 25000]


def ceil_div(a, k):
    return (a + k - 1) // k


def scene_oracle(scene):
    recs = []
    with (RUNS / scene / "trace" / "phase_a_records.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r["iteration"])
    out = {"scene": scene, "iters": [], "bucket_count": {}, "oracle": {},
           "tail": {}}
    per_iter = {}
    for r in recs:
        it = r["iteration"]
        npz = RUNS / scene / "trace" / f"phase_a_raw_{it:06d}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        ir = d["instance_ranges"].astype(np.int64)
        S = (ir[:, 1] - ir[:, 0])
        S = S[S > 0]
        bc = {int(k): int(ceil_div(S, k).sum()) for k in KS}
        # per-tile adaptive oracle: each tile picks K minimizing its bucket count
        # (bucket count is the only K-dependent structural cost); oracle = min over K
        oracle_buckets = int(min(ceil_div(S, k).sum() for k in KS))
        # short-tail occupancy at K=32: valid splats in final bucket per tile
        K = 32
        last_bucket_valid = (S % K)
        last_bucket_valid = np.where(last_bucket_valid == 0, K, last_bucket_valid)
        frac_le8 = float(np.mean(last_bucket_valid <= 8))
        frac_le16 = float(np.mean(last_bucket_valid <= 16))
        frac_le24 = float(np.mean(last_bucket_valid <= 24))
        per_iter[it] = {
            "n_tiles_with_splats": int(S.size),
            "bucket_count": bc,
            "oracle_buckets": oracle_buckets,
            "bucket_reduction_vs_K32": 1.0 - oracle_buckets / bc[32],
            "K64_reduction_vs_K32": 1.0 - bc[64] / bc[32],
            "tail_frac_le8": frac_le8, "tail_frac_le16": frac_le16,
            "tail_frac_le24": frac_le24,
            "splat_p50": float(np.percentile(S, 50)),
            "splat_p90": float(np.percentile(S, 90)),
            "splat_p99": float(np.percentile(S, 99)),
            "splat_max": int(S.max()),
        }
    out["per_iter"] = per_iter
    out["iters"] = sorted(per_iter.keys())
    return out


result = {}
for s in SCENES:
    result[s] = scene_oracle(s)
    print("===", s, "===")
    for it, v in result[s]["per_iter"].items():
        print("  iter=%6d tiles=%d bc32=%d bc64=%d oracle=%d K64_red=%.3f oracle_red=%.3f tail<=8=%.3f splat_p50=%.0f p99=%.0f max=%d" % (
            it, v["n_tiles_with_splats"], v["bucket_count"][32], v["bucket_count"][64],
            v["oracle_buckets"], v["K64_reduction_vs_K32"], v["bucket_reduction_vs_K32"],
            v["tail_frac_le8"], v["splat_p50"], v["splat_p99"], v["splat_max"]))

(RUNS.parent / "artifacts" / "DSH7_BARS0").mkdir(parents=True, exist_ok=True)
(RUNS.parent / "artifacts" / "DSH7_BARS0" / "BARS0_STRUCTURAL_ORACLE.json").write_text(
    json.dumps(result, indent=2))
print("wrote BARS0_STRUCTURAL_ORACLE.json")
