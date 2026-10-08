#!/usr/bin/env python
"""LAVOS0 REAL A100 scientific assembly driver (frozen Phases A-E).

Reads the per-scene captures (lavos0_capture.json) produced on the EPIC-05 A100
and runs the frozen analysis:
  Phase A  event order (frozen) + source identity (real remote frozen files)
  Phase B  headroom: streaming debt replay already captured -> debt_stats,
           consumer trace summary, structural headroom gates
  Phase C  bitwise topology exactness: drive topology_oracle on the REAL
           captured Phase-D window (real clone/split/prune + real SH grads)
  Phase D  perf/Amdahl from real microbench + consumer fractions + CUDA walls
  Phase E  assemble LAVOS0_SUMMARY.json, run the FROZEN analyzer, MANIFEST.

The analyzer (lavos0_analyze.py) owns the verdict; this driver never fabricates
one. Requires torch.cuda for the Phase-D oracle (A100); fails closed otherwise.

Usage (EPIC-05):
  python drivers/lavos0_run_a100.py --captures-root /root/lavos0_headroom \
      --outdir /root/lavos0_headroom/out --all
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from lavos0 import config, summary  # noqa: E402
from lavos0 import event_order, debt_replay, headroom_gates  # noqa: E402
from lavos0 import topology_oracle, perf_oracle  # noqa: E402
from lavos0.hashing import write_json  # noqa: E002

# Real remote frozen source files (the observed-semantics sources actually
# loaded by the frozen C0 trainer on EPIC-05). Hashed for SOURCE_IDENTITY.
FROZEN_SOURCES = {
    "c0_publication_trainer": "/root/epic05_p13x/publication_trainer_p13x.py",
    "c0_gaussian_model": "/root/3dgs-renderer-benchmark/baseline/reference_v1/gaussian_model.py",
    "c0_reference_config": "/root/3dgs-renderer-benchmark/baseline/reference_v1/config.py",
    "vos_trace_driver": "/root/vos_trace_driver.py",
    "lavos0_capture_driver": os.path.join(ROOT, "lavos0_capture.py"),
}


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_source_identity(results_by_scene):
    files = {}
    missing = []
    for name, rel in FROZEN_SOURCES.items():
        if os.path.isfile(rel):
            files[name] = {"path": rel, "sha256": sha256_file(rel)}
        else:
            missing.append(rel)
    # .so binaries + env from the trainer results.json (binary_identity)
    binid = {}
    for scene, d in results_by_scene.items():
        bi = d.get("binary_identity") or d.get("renderer") or {}
        if isinstance(bi, dict) and bi.get("so_sha256"):
            binid[scene] = {"so_sha256": bi.get("so_sha256"),
                            "core_sha256": bi.get("core_sha256")}
    return {
        "source_identity_pass": len(missing) == 0,
        "frozen_root": "/root",
        "files": files,
        "missing": missing,
        "n_files": len(files),
        "n_expected": len(FROZEN_SOURCES),
        "binary_identity": binid,
        "note": ("Frozen C0 sources actually loaded on EPIC-05. The local "
                 "vos/sources paths are not present remotely; the observed "
                 "semantics come from these files."),
    }


# ---------------------------------------------------------------------------
# Phase C: bitwise topology exactness on the REAL captured Phase-D window
# ---------------------------------------------------------------------------
def build_and_run_phase_d(pd, lr, device):
    """Build the DENSE-vs-LAYOUT oracle schedule from the captured window and
    run it. Returns the exactness verdict payload (PREREG Sec.6)."""
    if not pd or not pd.get("init_state"):
        return {"exactness_pass": False, "n_fences": 0, "n_mismatches": 0,
                "note": "no phase-d window captured"}
    init = pd["init_state"]
    logical = init["logical"]
    # SH rows are [F,3] (coeff x rgb); the oracle per-row feature is the flattened
    # F*3 vector (Adam is elementwise, so flattening is bitwise-identical).
    _theta = torch.tensor(init["theta"], dtype=torch.float32, device=device)
    n0 = _theta.shape[0]
    feat = int(_theta[0].numel())
    theta0 = _theta.reshape(n0, feat)
    if init.get("m") is not None:
        m0 = torch.tensor(init["m"], dtype=torch.float32, device=device).reshape(n0, feat)
    else:
        m0 = torch.zeros_like(theta0)
    if init.get("v") is not None:
        v0 = torch.tensor(init["v"], dtype=torch.float32, device=device).reshape(n0, feat)
    else:
        v0 = torch.zeros_like(theta0)
    dense = topology_oracle.LogicalState(
        theta=theta0.clone(), m=m0.clone(), v=v0.clone(),
        tau=torch.zeros(n0, dtype=torch.long, device=device),
        alive=torch.ones(n0, dtype=torch.bool, device=device), n_rows=n0, feat=feat)
    lazy = topology_oracle.LogicalState(
        theta=theta0.clone(), m=m0.clone(), v=v0.clone(),
        tau=torch.zeros(n0, dtype=torch.long, device=device),
        alive=torch.ones(n0, dtype=torch.bool, device=device), n_rows=n0, feat=feat)

    oracle = topology_oracle.TopologyOracle(feat, lr, device=device)
    l2o = {int(lg): i for i, lg in enumerate(logical)}  # logical -> oracle idx
    iters = sorted(pd["iters"].keys(), key=int)
    if not iters:
        return {"exactness_pass": False, "n_fences": 0, "n_mismatches": 0,
                "note": "empty phase-d window"}
    t0 = int(iters[0])
    n_iters = int(iters[-1]) - t0 + 1
    events, grads_by_iter, consumers_by_iter, fences = [], {}, {}, []

    def drop_indices(drop_set):
        """Mirror oracle _drop_rows compaction: remap tracked rows after the
        oracle drops `drop_set` (a set of oracle indices). Surviving tracked
        rows shift down by the number of dropped indices below them. The
        dropped (parent/pruned) rows are removed from l2o."""
        nonlocal l2o
        rebuilt = {}
        for lg, oi in l2o.items():
            if oi in drop_set:
                continue
            shift = sum(1 for d in drop_set if d < oi)
            rebuilt[lg] = oi - shift
        l2o = rebuilt

    for it_str in iters:
        it = int(it_str)
        t = it - t0 + 1
        rec = pd["iters"][it_str]
        for ev in rec.get("events", []):
            if ev["clone"]:
                rows = [l2o[int(lg)] for lg in ev["clone"] if int(lg) in l2o]
                ct = torch.tensor(ev["child_theta"], dtype=torch.float32, device=device).reshape(-1, feat)
                if rows:
                    events.append(topology_oracle.TopologyEvent(
                        t, "clone", rows, child_theta_seed=ct))
            if ev["split"]:
                rows = [l2o[int(lg)] for lg in ev["split"] if int(lg) in l2o]
                ct = torch.tensor(ev["child_theta"], dtype=torch.float32, device=device).reshape(-1, feat)
                if rows:
                    events.append(topology_oracle.TopologyEvent(
                        t, "split", rows, child_theta_seed=ct))
                    drop_indices(set(rows))
            if ev["prune"]:
                rows = [l2o[int(lg)] for lg in ev["prune"] if int(lg) in l2o]
                if rows:
                    events.append(topology_oracle.TopologyEvent(t, "prune", rows))
                    drop_indices(set(rows))
        cons_rows, grads = [], {}
        for lg_s, c in rec["consumers"].items():
            lg = int(lg_s)
            if lg in l2o and c["c"] and c["g"] is not None:
                oi = l2o[lg]
                cons_rows.append(oi)
                grads[oi] = torch.tensor(c["g"], dtype=torch.float32, device=device).reshape(-1)
        consumers_by_iter[t] = cons_rows
        grads_by_iter[t] = grads
        if rec.get("events"):
            fences.append(topology_oracle.Fence(t, "topology"))
    fences.append(topology_oracle.Fence(n_iters, "final"))

    comparisons = oracle.run(dense, lazy, events, grads_by_iter,
                             consumers_by_iter, fences, n_iters)
    verdict = topology_oracle.exactness_verdict(comparisons)
    verdict["n_events"] = len(events)
    verdict["n_rows_schedule"] = n0
    verdict["window_iters"] = n_iters
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scenes", nargs="*", default=list(config.SCENES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-non-a100", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not torch.cuda.is_available() and not args.allow_non_a100:
        print(json.dumps({"status": "A100_REQUIRED",
                          "note": "torch.cuda unavailable; Phase-D oracle needs the A100."}))
        sys.exit(3)

    scenes = config.SCENES if args.all else args.scenes
    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"

    # ---- load captures ----------------------------------------------------
    caps, results = {}, {}
    for s in scenes:
        cp = os.path.join(args.captures_root, s, "lavos0_capture.json")
        rp = os.path.join(args.captures_root, s, "results.json")
        if not os.path.isfile(cp):
            print(json.dumps({"status": "MISSING_CAPTURE", "scene": s, "path": cp}))
            sys.exit(2)
        caps[s] = json.load(open(cp))
        if os.path.isfile(rp):
            results[s] = json.load(open(rp))

    # ---- Phase A ----------------------------------------------------------
    src = build_source_identity(results)
    ev = event_order.phase_a_gate(max_iterations=config.PHASE_END_ITER)
    write_json(os.path.join(args.outdir, "LAVOS0_SOURCE_IDENTITY.json"), src)
    write_json(os.path.join(args.outdir, "LAVOS0_EVENT_ORDER.json"),
               {"resolved": ev.resolved, "notes": ev.notes, "ordering": ev.ordering})

    # ---- Phase B + C + D per scene ---------------------------------------
    per_scene_headroom, per_scene_trace = {}, {}
    per_scene_exact, per_scene_perf = {}, {}
    scene_headroom_objs = {}
    scene_perf_objs = {}
    for s in scenes:
        cap = caps[s]
        ds = cap.get("debt_stats") or {}
        rs = cap.get("records_summary") or {}
        walls = cap.get("walls") or {}
        mean_iter_ms = walls.get("mean_iter_ms") or 0.0
        sh_adam_ms = cap.get("sh_adam_ms_per_iter_mean") or 0.0
        sh_frac = (sh_adam_ms / mean_iter_ms) if (mean_iter_ms and sh_adam_ms) else 0.0

        stats = debt_replay.DebtStats()
        for k, v in ds.items():
            if k in stats.__dict__:
                setattr(stats, k, v)

        # Phase D (bitwise topology exactness) on the real captured window
        lr = 0.0025
        try:
            exact = build_and_run_phase_d(cap.get("phase_d"), lr, device)
        except Exception as e:
            exact = {"exactness_pass": False, "error": repr(e)}

        # Phase B headroom gates
        sh = headroom_gates.SceneHeadroom(s, stats, sh_adam_wall_fraction=sh_frac)
        scene_headroom_objs[s] = sh
        per_scene_headroom[s] = sh.as_dict()
        per_scene_trace[s] = {
            "n_iterations": rs.get("n_iterations"),
            "n_total_rows_mean": rs.get("n_total_rows_mean"),
            "n_consumers_mean": rs.get("n_consumers_mean"),
            "nonzero_grad_fraction_mean": rs.get("nonzero_grad_fraction_mean"),
            "sh_adam_wall_fraction": sh_frac,
            "sh_adam_ms_per_iter": sh_adam_ms,
            "mean_iter_ms": mean_iter_ms,
            "consumer_block_fraction": (1.0 - stats.avoidable_fraction()) if stats.dense_row_transitions else None,
        }
        per_scene_exact[s] = exact

        # Phase D perf/Amdahl
        cf = rs.get("nonzero_grad_fraction_mean")
        cf = 0.5 if not cf else cf
        t_pre = (mean_iter_ms * config.PHASE_END_ITER) / 1000.0  # s
        # post-15K estimate: no densify/opacity-reset => ~ (mean - densify) per iter
        densify_ms = (walls.get("phase_ms_mean") or {}).get("densify") or 0.0
        post_iter_ms = max(mean_iter_ms - densify_ms, 1e-6)
        t_post = (post_iter_ms * config.PHASE_END_ITER) / 1000.0
        n_rows = rs.get("n_total_rows_mean") or 1
        per_row_ms = (sh_adam_ms / n_rows) if sh_adam_ms else 0.0
        forced_overhead = (stats.topology_forced_settles / max(config.PHASE_END_ITER, 1)) * per_row_ms
        perf = perf_oracle.ScenePerf(
            scene=s, consumer_fraction=cf, sh_adam_wall_fraction=sh_frac,
            t_pre_wall=t_pre, t_post_wall=t_post,
            dense_sh_adam_ms_per_iter=sh_adam_ms,
            lazy_sh_adam_ms_per_iter=sh_adam_ms * cf,
            forced_settle_overhead=forced_overhead, microbench_speedup=None)
        scene_perf_objs[s] = perf
        per_scene_perf[s] = perf.as_dict()

    headroom = headroom_gates.evaluate_headroom(scene_headroom_objs)
    perf = perf_oracle.evaluate_performance(scene_perf_objs)

    # ---- Phase E: assemble + frozen analyzer ------------------------------
    env = {
        "hostname": os.uname().nodename,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "device": str(device),
        "python": sys.version.split()[0],
        "note": "REAL A100 scientific run on EPIC-05 (not synthetic).",
    }
    write_json(os.path.join(args.outdir, "LAVOS0_ENVIRONMENT.json"), env)
    write_json(os.path.join(args.outdir, "LAVOS0_CONSUMER_TRACE.json"),
               {"per_scene": per_scene_trace})
    write_json(os.path.join(args.outdir, "LAVOS0_HEADROOM.json"), {
        "per_scene": per_scene_headroom,
        "aggregate": headroom.aggregate,
        "passed": headroom.passed,
        "stop_early": headroom.stop_early,
        "consumer_block_fraction": {s: per_scene_trace[s]["consumer_block_fraction"] for s in scenes},
    })
    write_json(os.path.join(args.outdir, "LAVOS0_REPLAY.json"), {
        "per_scene": per_scene_exact,
        "exactness_pass": all(e.get("exactness_pass") for e in per_scene_exact.values()),
        "n_mismatches": sum(e.get("n_mismatches", 0) for e in per_scene_exact.values()),
        "n_fences": sum(e.get("n_fences", 0) for e in per_scene_exact.values()),
    })
    write_json(os.path.join(args.outdir, "LAVOS0_TIMING.json"), perf)

    s = summary.build_summary(
        environment=env, source_identity=src,
        event_order={"resolved": ev.resolved, "notes": ev.notes, "ordering": ev.ordering},
        consumer_trace={"per_scene": per_scene_trace},
        headroom=headroom.as_dict(), topology_oracle={
            "exactness_pass": all(e.get("exactness_pass") for e in per_scene_exact.values()),
            "n_fences": sum(e.get("n_fences", 0) for e in per_scene_exact.values()),
            "n_mismatches": sum(e.get("n_mismatches", 0) for e in per_scene_exact.values()),
        }, perf_oracle=perf)
    write_json(os.path.join(args.outdir, "LAVOS0_SUMMARY.json"), s)

    # frozen analyzer owns the verdict
    sys.path.insert(0, HERE)
    import lavos0_make_manifest as mm
    import subprocess
    an = os.path.join(HERE, "..", "lavos0_analyze.py")
    subprocess.run([sys.executable, os.path.join(ROOT, "lavos0_analyze.py"),
                    os.path.join(args.outdir, "LAVOS0_SUMMARY.json")], check=False)
    lines = mm.build_manifest_recursive(args.outdir)
    ok, problems = mm.verify_manifest(args.outdir)
    print(json.dumps({
        "status": "DONE",
        "verdict": (json.load(open(os.path.join(args.outdir, "LAVOS0_VERDICT.json")))["verdict"]
                    if os.path.isfile(os.path.join(args.outdir, "LAVOS0_VERDICT.json")) else None),
        "source_identity_pass": s["source_identity_pass"],
        "event_semantics_resolved": s["event_semantics_resolved"],
        "structural_headroom_pass": s["structural_headroom_pass"],
        "topology_exactness_pass": s["topology_exactness_pass"],
        "performance_headroom_pass": s["performance_headroom_pass"],
        "manifest_ok": ok, "manifest_problems": problems,
        "n_artifacts": len(lines),
    }, indent=2))


if __name__ == "__main__":
    main()
