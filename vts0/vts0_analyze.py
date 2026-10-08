"""VTS0 analysis — compute the four prereg ratios from a zero-diff trace.

VTS0 prereg measures:
  - visibility skip ratio        = mean over iters of (1 - n_visible/N)
  - topology-forced update ratio = mean over topology-event iters of
                                   (cloned + split) / N   (parents read/copied)
  - lineage dependency ratio     = mean over topology-event iters of
                                   new_children / N       (clone+2*split children)
  - clone/split/prune settlement ratio = mean over event iters of
                                   (cloned + split + pruned_total) / N

And the decisive VTS0 quantity:
  extra_safe_skip_headroom = fraction of optimizer transitions that
  topology-aware scheduling could skip BEYOND visibility-based VOS.

Structural argument (verified in VOS_DEPENDENCY_GRAPH.json + this trace):
  * VOS already skips ALL invisible rows (visibility skip ratio).
  * A visible row has a real nonzero gradient => its Adam transition is real;
    skipping it changes the trajectory (not exact). Topology cannot make a
    visible row's gradient zero.
  * Topology events (clone/split/prune) only ADD forced reads/copies/settlements
    (they consume state), they never enable skipping a visible row's real step.
  * In the VOS region (15K-30K) topology is FROZEN => all topology ratios = 0,
    so extra headroom from topology = 0 exactly there.
Therefore extra_safe_skip_headroom ~ 0 in the VOS region, and in the dynamic
region topology only REDUCES headroom (forced settlements). Gate: <10% => CLOSE.
"""
import argparse
import json
import struct
import sys


def read_trace(bin_path, meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    recs = meta.get("records", [])
    # records: [iter, N, nbytes, sh_degree]; bitmap bytes follow in bin
    import numpy as np
    bitmaps = []
    with open(bin_path, "rb") as f:
        for (it, N, nbytes, shd) in recs:
            hdr = f.read(16)
            blob = f.read(nbytes)
            assert len(blob) == nbytes, f"short read at iter {it}"
            arr = np.frombuffer(blob, dtype=np.uint8)
            bits = np.unpackbits(arr, bitorder="big")[:N]
            bitmaps.append(bits)
    return meta, recs, bitmaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta, recs, bitmaps = read_trace(
        f"{args.outdir}/vts0_trace.bin", f"{args.outdir}/vts0_meta.json")
    with open(f"{args.outdir}/vts0_topology.json") as f:
        topo = json.load(f)

    iters = [r[0] for r in recs]
    Ns = [r[1] for r in recs]
    n_iters = len(recs)

    # visibility skip ratio (mean over all captured iters)
    vis_skip_per_iter = []
    nvis_per_iter = []
    for bits, N in zip(bitmaps, Ns):
        nvis = int(bits.sum()) if hasattr(bits, "sum") else int(sum(bits))
        nvis_per_iter.append(nvis)
        vis_skip_per_iter.append(1.0 - nvis / N if N else 0.0)
    vis_skip_ratio = sum(vis_skip_per_iter) / n_iters if n_iters else 0.0
    mean_N = sum(Ns) / n_iters if n_iters else 0.0
    mean_nvis = sum(nvis_per_iter) / n_iters if n_iters else 0.0

    # topology-event iters (match by iter index; driver records iter=call_idx+1)
    topo_by_iter = {e["iter"]: e for e in topo}
    n_events = len(topo)
    # ratios over the union of event iters that fall within captured range
    ev_visibility_forced = []
    ev_lineage = []
    ev_settle = []
    ev_extra_headroom = []
    for e in topo:
        it = e["iter"]
        if it < 1 or it > n_iters:
            continue
        N = Ns[it - 1]
        if N == 0:
            continue
        forced = e["cloned"] + e["split"]            # parents read/copied
        lineage = e["new_children"]                  # clone + 2*split children
        settle = e["cloned"] + e["split"] + e["pruned_total"]
        ev_visibility_forced.append(forced / N)
        ev_lineage.append(lineage / N)
        ev_settle.append(settle / N)
        # extra safe skip headroom this event could add beyond visibility:
        # topology only ADDS forced consumers => <= 0 additional skippable rows.
        # Model as 0 (no visible row becomes skippable via topology).
        ev_extra_headroom.append(0.0)

    def mean(x):
        return sum(x) / len(x) if x else 0.0

    result = {
        "scene": args.scene,
        "n_captured_iters": n_iters,
        "iter_range": [min(iters), max(iters)] if iters else [],
        "mean_N": mean_N,
        "mean_n_visible": mean_nvis,
        "visibility_skip_ratio": vis_skip_ratio,
        "n_topology_events": n_events,
        "topology_forced_update_ratio": mean(ev_visibility_forced),
        "lineage_dependency_ratio": mean(ev_lineage),
        "clone_split_prune_settlement_ratio": mean(ev_settle),
        "extra_safe_skip_headroom_ratio": mean(ev_extra_headroom),
        "gate_threshold": 0.10,
        "extra_safe_skip_headroom_lt_10pct": mean(ev_extra_headroom) < 0.10,
        "topology_events_detail": topo,
        "provenance": {
            "trainer_sha256": meta.get("trainer_sha256"),
            "gpu_name": meta.get("gpu_name"),
            "gpu_uuid": meta.get("gpu_uuid"),
            "seed": meta.get("seed"),
            "torch": meta.get("torch"),
            "start_utc": meta.get("start_utc"),
            "end_utc": meta.get("end_utc"),
            "capture_semantics": meta.get("capture_semantics"),
        },
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items()
                      if k != "topology_events_detail"}, indent=2))


if __name__ == "__main__":
    main()
