#!/usr/bin/env python3
"""P10 Gate H3 - direct backward microbenchmark, gsplat main H8-MR.

Protocol per (scene, arm)
  warmup  20 iterations of the full fwd+bwd step (not timed)
  100 iterations per repetition, 5 repetitions
  CUDA events per iteration around the backward region itself; the forward that
  produces the autograd graph runs before the start event, so the measured
  region is the direct backward (rasterizer bwd + projection bwd + autograd
  plumbing in between). The fwd+bwd region is measured as a whole as well.

  Primary quantity: BACKWARD-ONLY median over the 5 repetitions. Forward alone
  is reported for context and as a null check (the switch cannot move it).

Clean-GPU precondition: refuses to run when another compute process is resident
on the target GPU (the host runs a publication campaign on GPUs 0-3 and an
unrelated job on 4-7) unless --allow-shared is given. Recorded in the output.

modes
  bench --tag h8off --binary h8 --arm off --scene room --gpu 0 --out dumps/h3_room.json
  parse --scene-room dumps/h3_room.json ... --out EPIC05_P10_TIMING.json
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, "/root/epic05_p10")
from p10_h2_equivalence import BINARIES, build_inputs, load_arm  # noqa: E402

SEED = 4200


def gpu_state(gpu):
    try:
        info = subprocess.run(
            ["nvidia-smi", f"--id={gpu}",
             "--query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        info = f"unavailable: {e}"
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"query": info, "compute_apps": apps}


def resident_foreign(gpu):
    uuid = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    n = 0
    for line in subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1] == uuid:
            n += 1
    return n


def cmd_bench(a):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    if a.arm == "on":
        os.environ["GSPLAT_H8_MR"] = "1"
    else:
        os.environ.pop("GSPLAT_H8_MR", None)

    resident = resident_foreign(a.gpu)
    pre = {"gpu_index": a.gpu, "foreign_resident_compute_processes": resident,
           "allow_shared": a.allow_shared, "state": gpu_state(a.gpu)}
    if resident and not a.allow_shared:
        print(f"H3 {a.tag}: REFUSED - {resident} foreign compute process(es) resident on GPU {a.gpu}")
        sys.exit(3)

    import torch
    rasterization, load_ply, load_cams, resize_cams = load_arm(a.binary)
    params, cam, V = build_inputs(load_ply, load_cams, resize_cams, a.scene)
    n_gauss = int(params["means"].shape[0])

    def forward():
        for t in params.values():
            t.grad = None
        r, _, _ = rasterization(
            means=params["means"], quats=params["quats"], scales=params["scales"],
            opacities=params["opacities"], colors=params["sh"],
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=16, packed=False, sh_degree=3, radius_clip=0.0,
            eps2d=0.1, render_mode="RGB", absgrad=False)
        return r[0]

    def full():
        (forward() * V).sum().backward()

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(a.iters)]

    def time_region(with_bwd):
        """One event pair per iteration, preallocated; the region is the backward
        alone when with_bwd, the forward alone otherwise."""
        for s, e in events:
            render = forward()
            s.record()
            if with_bwd:
                (render * V).sum().backward()
            e.record()
        torch.cuda.synchronize()
        return statistics.fmean(s.elapsed_time(e) for s, e in events)

    for _ in range(a.warmup):
        full()
    torch.cuda.synchronize()

    ms = {"backward": [], "forward": [], "fwd_bwd": []}
    for _ in range(a.reps):
        ms["backward"].append(time_region(True))
        ms["forward"].append(time_region(False))
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(a.iters):
            full()
        e.record()
        torch.cuda.synchronize()
        ms["fwd_bwd"].append(s.elapsed_time(e) / a.iters)

    payload = {
        "tag": a.tag, "binary": a.binary, "arm": a.arm, "scene": a.scene,
        "n_gaussians": n_gauss, "width": cam.image_width, "height": cam.image_height,
        "seed": SEED,
        "protocol": {"warmup": a.warmup, "iters": a.iters, "reps": a.reps,
                     "timing": "one CUDA event pair per iteration inside the region, "
                               "mean per repetition, median over repetitions",
                     "backward_region": "the backward() call only; the forward producing the "
                                        "autograd graph is issued before the start event",
                     "primary": "backward"},
        "precondition": pre,
        "ms": {k: {"per_rep": v, "median": statistics.median(v), "mean": statistics.fmean(v),
                   "min": min(v), "max": max(v)} for k, v in ms.items()},
    }
    data = {"scene": a.scene, "binary": a.binary, "tags": {}}
    if os.path.exists(a.out):
        with open(a.out) as f:
            prev = json.load(f)
        if isinstance(prev, dict) and "tags" in prev and prev.get("scene") == a.scene:
            data = prev
    data["tags"][a.tag] = payload
    with open(a.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"H3 {a.tag} {a.scene}: bwd median={payload['ms']['backward']['median']:.4f} ms "
          f"fwd median={payload['ms']['forward']['median']:.4f} ms "
          f"fwd+bwd={payload['ms']['fwd_bwd']['median']:.4f} ms -> {a.out}")


def _gain_row(base, h8, key):
    import statistics as st
    b, h = base["ms"][key], h8["ms"][key]
    per = [(bi / hi - 1.0) * 100.0 for bi, hi in zip(b["per_rep"], h["per_rep"])]
    med = (b["median"] / h["median"] - 1.0) * 100.0
    ci = None
    if len(per) >= 3:
        # half-width of the 95% t interval on the paired per-repetition gains
        t95 = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
               10: 2.262}.get(len(per), 1.96)
        ci = t95 * st.stdev(per) / (len(per) ** 0.5)
    return {"base_median_ms": b["median"], "h8_median_ms": h["median"],
            "gain_pct_median": med, "gain_pct_per_rep": per, "gain_pct_ci95": ci}


def cmd_parse(a):
    scenes = [s for s in ("room", "bicycle", "garden") if getattr(a, f"scene_{s}")]
    out = {"scenes": {}, "protocol": {
               "warmup": 20, "iters": 100, "reps": 5,
               "regions": {"backward": "primary (direct backward)", "forward": "context",
                           "fwd_bwd": "context"},
               "ratio": "baseline_ms / h8_ms; > 1 means H8 faster"}}
    ratios = []
    for s in scenes:
        with open(getattr(a, f"scene_{s}")) as f:
            d = json.load(f)
        tags = sorted(d["tags"])
        base = next((t for t in tags if t.endswith("off")), tags[0])
        h8 = next((t for t in tags if t.endswith("on")), tags[-1])
        row = {"n_gaussians": d["tags"][base]["n_gaussians"],
               "arms": {t: {k: d["tags"][t]["ms"][k]["median"] for k in ("forward", "backward", "fwd_bwd")}
                        for t in tags},
               "per_rep": {t: {k: d["tags"][t]["ms"][k]["per_rep"] for k in ("forward", "backward", "fwd_bwd")}
                           for t in tags},
               "baseline_tag": base, "h8_tag": h8,
               "precondition": {t: d["tags"][t]["precondition"] for t in tags}}
        row["ratios"] = {k: row["arms"][base][k] / row["arms"][h8][k] for k in ("forward", "backward", "fwd_bwd")}
        row["h8_speedup_pct_backward"] = (row["ratios"]["backward"] - 1.0) * 100.0
        ratios.append(row["ratios"]["backward"])
        out["scenes"][s] = row

        # verdict-compatible record (schema p10_verdict.py reads)
        vrec = {"scene": s, "n_gaussians": row["n_gaussians"],
                "direct_backward": {"off": {"median_ms": row["arms"][base]["backward"],
                                            "per_rep": row["per_rep"][base]["backward"]},
                                    "on": {"median_ms": row["arms"][h8]["backward"],
                                           "per_rep": row["per_rep"][h8]["backward"]},
                                    **{k: v for k, v in _gain_row(d["tags"][base], d["tags"][h8], "backward").items()
                                       if k.startswith("gain")}},
                "forward_backward": {"off": {"median_ms": row["arms"][base]["fwd_bwd"],
                                             "per_rep": row["per_rep"][base]["fwd_bwd"]},
                                     "on": {"median_ms": row["arms"][h8]["fwd_bwd"],
                                            "per_rep": row["per_rep"][h8]["fwd_bwd"]},
                                     **{k: v for k, v in _gain_row(d["tags"][base], d["tags"][h8], "fwd_bwd").items()
                                        if k.startswith("gain")}},
                "forward_only": {"off_ms": row["arms"][base]["forward"],
                                 "on_ms": row["arms"][h8]["forward"],
                                 "note": "null check: the switch cannot move the forward pass"},
                "precondition": row["precondition"]}
        pri = next((t for t in tags if t == "pri"), None)
        if pri:
            vrec["pristine_crosscheck"] = {"direct_backward_base_ms": d["tags"][pri]["ms"]["backward"]["median"],
                                           "direct_backward_base_per_rep": d["tags"][pri]["ms"]["backward"]["per_rep"],
                                           "patched_off_over_pristine_pct":
                                               (row["arms"][base]["backward"]
                                                / d["tags"][pri]["ms"]["backward"]["median"] - 1.0) * 100.0}
        outdir = os.path.dirname(os.path.abspath(getattr(a, f"scene_{s}"))) or "."
        with open(os.path.join(outdir, f"h3_verdict_{s}.json"), "w") as f:
            json.dump(vrec, f, indent=2)
        if pri:
            # the shape p10_verdict.py expects for the pristine cross-build arm
            with open(os.path.join(outdir, f"h3_verdict_{s}_pri.json"), "w") as f:
                json.dump({"scene": s, "n_gaussians": row["n_gaussians"],
                           "direct_backward": {"base": {
                               "median_ms": d["tags"][pri]["ms"]["backward"]["median"],
                               "per_rep": d["tags"][pri]["ms"]["backward"]["per_rep"]}},
                           "forward_backward": {"base": {
                               "median_ms": d["tags"][pri]["ms"]["fwd_bwd"]["median"]}}}, f, indent=2)
    if ratios:
        import math
        geom = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        out["geomean_backward_ratio"] = geom
        out["geomean_backward_speedup_pct"] = (geom - 1.0) * 100.0
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    for s, r in out["scenes"].items():
        print(f"{s}: bwd ratio={r['ratios']['backward']:.4f} ({r['h8_speedup_pct_backward']:+.2f}%) "
              f"fwd={r['ratios']['forward']:.4f} fwd+bwd={r['ratios']['fwd_bwd']:.4f}")
    if ratios:
        print(f"GEOMEAN backward ratio={out['geomean_backward_ratio']:.4f} "
              f"({out['geomean_backward_speedup_pct']:+.2f}%)")
    print(f"TIMING -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--tag", required=True)
    b.add_argument("--binary", choices=sorted(BINARIES), default="h8")
    b.add_argument("--arm", choices=["off", "on"], required=True)
    b.add_argument("--scene", required=True)
    b.add_argument("--gpu", type=int, default=0)
    b.add_argument("--warmup", type=int, default=20)
    b.add_argument("--iters", type=int, default=100)
    b.add_argument("--reps", type=int, default=5)
    b.add_argument("--allow-shared", action="store_true")
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_bench)
    p = sub.add_parser("parse")
    for s in ("room", "bicycle", "garden"):
        p.add_argument(f"--scene-{s}", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_parse)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
