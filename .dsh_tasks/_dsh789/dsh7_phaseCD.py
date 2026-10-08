#!/usr/bin/env python3
"""DSH-7 BARS0 Phase C (adaptive oracle) + Phase D (tail packing) generator.

Structural oracle computed from Phase-A per-tile splat counts (exact, from the
raw npz instance_ranges). The per-tile adaptive oracle chooses K_t in {8,16,32,64}
minimizing predicted cost = checkpoint_store_cost + backward_replay_cost. Both
components scale with the tile's bucket count ceil(S_t/K), which is monotone
non-increasing in K, so the oracle selects K_t=64 for every tile. This is a
STRUCTURAL result (bucket-count reduction); conversion to renderer-time savings
(B1) requires the Phase B static-K microbench, which is reported separately.
"""
import json
from pathlib import Path
import numpy as np

RUNS = Path("/root/dsh_20261001/runs/DSH_PHASEA")
SCENES = ["room", "garden", "bicycle"]
KS = [8, 16, 32, 64]
MATURE = [10000, 14500, 20000, 25000]
BLOCK = 256  # pixels per tile


def ceil_div(a, k):
    return (a + k - 1) // k


def main():
    out_root = RUNS.parent / "artifacts" / "DSH7_BARS0"
    out_root.mkdir(parents=True, exist_ok=True)

    oracle = {
        "p0": "DSH-7 BARS0 adaptive-K oracle (Phase C)",
        "frozen_objective": "minimize per-tile (checkpoint_store_cost + "
                            "backward_replay_cost); both scale with "
                            "bucket_count = ceil(S_t / K)",
        "K_set": KS,
        "K_set_frozen": True,
        "structural_finding": "bucket_count is monotone non-increasing in K, "
                              "so the per-tile oracle selects K_t=64 for EVERY "
                              "tile. 'Adaptive K' collapses to global K=64; "
                              "per-tile adaptivity adds no benefit over K=64.",
        "scenes": {},
    }
    tail = {
        "p0": "DSH-7 BARS0 short-tail warp-packing (Phase D)",
        "K": 32,
        "note": "Final-bucket valid-splat occupancy per tile at K=32. A tile "
                "with S_t splats has a final bucket with S_t % 32 (or 32 if "
                "divisible) valid splats; the remaining lanes are idle.",
        "scenes": {},
    }

    for s in SCENES:
        recs = []
        p = RUNS / s / "trace" / "phase_a_records.jsonl"
        if p.exists():
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
        recs.sort(key=lambda r: r["iteration"])
        per_iter = {}
        tail_per_iter = {}
        for r in recs:
            it = r["iteration"]
            npz = RUNS / s / "trace" / f"phase_a_raw_{it:06d}.npz"
            if not npz.exists():
                continue
            d = np.load(npz)
            ir = d["instance_ranges"].astype(np.int64)
            S = (ir[:, 1] - ir[:, 0])
            S = S[S > 0]
            if S.size == 0:
                continue
            bc = {int(k): int(ceil_div(S, k).sum()) for k in KS}
            # oracle: per-tile min bucket count over K (== K=64 since monotone)
            oracle_per_tile = np.min(np.stack([ceil_div(S, k) for k in KS], axis=1), axis=1)
            oracle_buckets = int(oracle_per_tile.sum())
            # tail occupancy at K=32
            last = S % 32
            last = np.where(last == 0, 32, last)
            tail_per_iter[it] = {
                "n_tiles": int(S.size),
                "frac_final_bucket_le8": float(np.mean(last <= 8)),
                "frac_final_bucket_le16": float(np.mean(last <= 16)),
                "frac_final_bucket_le24": float(np.mean(last <= 24)),
                "mean_final_bucket_valid": float(last.mean()),
                "mean_splat_p50": float(np.percentile(S, 50)),
                "mean_splat_p90": float(np.percentile(S, 90)),
                "mean_splat_p99": float(np.percentile(S, 99)),
                "splat_max": int(S.max()),
            }
            per_iter[it] = {
                "n_tiles": int(S.size),
                "bucket_count": bc,
                "oracle_buckets": oracle_buckets,
                "oracle_reduction_vs_K32": 1.0 - oracle_buckets / bc[32],
                "K64_reduction_vs_K32": 1.0 - bc[64] / bc[32],
                "checkpoint_bytes_K32": bc[32] * BLOCK * 16,
                "checkpoint_bytes_K64": bc[64] * BLOCK * 16,
            }
        oracle["scenes"][s] = {"n_iter": len(per_iter), "per_iter": per_iter}
        tail["scenes"][s] = {"n_iter": len(tail_per_iter), "per_iter": tail_per_iter}

    (out_root / "BARS0_ADAPTIVE_ORACLE.json").write_text(json.dumps(oracle, indent=2))
    (out_root / "BARS0_TAIL_PACKING.json").write_text(json.dumps(tail, indent=2))
    print("wrote BARS0_ADAPTIVE_ORACLE.json and BARS0_TAIL_PACKING.json")
    for s in SCENES:
        pi = oracle["scenes"][s]["per_iter"]
        if pi:
            last_it = max(pi.keys())
            v = pi[last_it]
            print(f"  {s} iter={last_it}: bc32={v['bucket_count'][32]} bc64={v['bucket_count'][64]} "
                  f"oracle_red={v['oracle_reduction_vs_K32']:.3f} ckpt_K32_MB={v['checkpoint_bytes_K32']/1e6:.0f}")


if __name__ == "__main__":
    main()
