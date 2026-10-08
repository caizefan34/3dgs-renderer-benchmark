#!/usr/bin/env python
"""Offline end-to-end LAVOS0 pipeline on SYNTHETIC (clearly-labeled) data.

This driver exercises the ENTIRE LAVOS0 artifact pipeline - Phase A event
order, Phase B consumer/debt trace, Phase C headroom gates, Phase D exactness
oracle, Phase E perf/Amdahl, summary assembly, the frozen analyzer, and the
MANIFEST - using a synthetic stochastic model of a 0..15K trace. It requires NO
GPU and fabricates NO scientific verdict: the emitted artifacts are labelled
`synthetic=true` and the REPORT states the numbers are placeholders pending the
real A100 run.

Purpose: prove the harness plumbing is correct end-to-end and that the frozen
analyzer + manifest integration works, independent of EPIC-05. The real
scientific run is drivers/lavos0_run_a100.py.

Usage:
    python drivers/lavos0_orchestrate_synthetic.py --outdir a100_out/synthetic
    python drivers/lavos0_orchestrate_synthetic.py --iters 15000   # slower
"""

import argparse
import dataclasses
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from lavos0 import config, summary                      # noqa: E402
from lavos0.hashing import write_json                  # noqa: E402
from lavos0 import event_order, consumer_trace, debt_replay  # noqa: E402
from lavos0 import headroom_gates, topology_oracle, perf_oracle  # noqa: E402
from lavos0 import source_identity                       # noqa: E402

import torch                                             # noqa: E402


def _synthetic_trace(scene, seed, n_iters=1500):
    """Generate a synthetic consumer trace + topology events for one scene.

    Models a growing SH population, a per-iteration consumer fraction that
    decays as the population grows (typical 3DGS), periodic topology events
    (clone/split/prune) and eval fences. Clearly synthetic.
    """
    g = torch.Generator().manual_seed(seed)
    n_initial = 800
    consumers = {}
    clone_p, split_p, prune_p = {}, {}, {}
    n = n_initial
    for t in range(1, n_iters + 1):
        frac = 0.70 - 0.35 * (t / n_iters)
        n_consumers = max(1, int(frac * n))
        rows = torch.randperm(n, generator=g)[:n_consumers].tolist()
        consumers[t] = sorted(rows)
        if t < n_iters and t % 100 == 0 and 100 < t < n_iters:
            n_clone = int(0.3 * n_consumers)
            n_split = int(0.2 * n_consumers)
            clone_ids = rows[:n_clone]
            split_ids = rows[n_clone:n_clone + n_split]
            if clone_ids:
                clone_p[t] = clone_ids
                n += len(clone_ids)
            if split_ids:
                split_p[t] = split_ids
                n += 2 * len(split_ids)   # children
                n -= len(split_ids)       # parents removed
            prune_ids = [r for r in range(min(5, n))]
            if prune_ids:
                prune_p[t] = prune_ids
                n -= len(prune_ids)
    return consumers, clone_p, split_p, prune_p, n_initial


def _run_synthetic_exactness(scene, seed):
    """Run the real Phase D oracle on a small synthetic clone/split/prune mix."""
    g = torch.Generator().manual_seed(seed + 1000)
    feat = 16
    n = 12
    theta = torch.randn((n, feat), generator=g)
    dense = topology_oracle.LogicalState(
        theta=theta, m=torch.randn((n, feat), generator=g) * 1e-2,
        v=torch.rand((n, feat), generator=g) + 0.5,
        tau=torch.zeros(n, dtype=torch.long),
        alive=torch.ones(n, dtype=torch.bool), n_rows=n, feat=feat)
    lazy = topology_oracle.LogicalState(
        theta=dense.theta.clone(), m=dense.m.clone(), v=dense.v.clone(),
        tau=dense.tau.clone(), alive=dense.alive.clone(),
        n_rows=n, feat=feat)
    oracle = topology_oracle.TopologyOracle(feat, lr=0.0025)
    grads = {1: {0: torch.randn(feat, generator=g)},
             2: {1: torch.randn(feat, generator=g)}}
    cons = {1: [0], 2: [1]}
    events = [
        topology_oracle.TopologyEvent(1, "clone", [2], dense.theta[2:3].clone()),
        topology_oracle.TopologyEvent(2, "split", [3],
                                      dense.theta[3:4].repeat(2, 1).clone()),
        topology_oracle.TopologyEvent(3, "prune", [5]),
    ]
    fences = [topology_oracle.Fence(1, "topology"),
              topology_oracle.Fence(2, "+1"),
              topology_oracle.Fence(3, "topology"),
              topology_oracle.Fence(4, "+8")]
    comp = oracle.run(dense, lazy, events, grads, cons, fences, n_iters=4)
    v = topology_oracle.exactness_verdict(comp)
    return {"scene": scene, "synthetic": True, **v}


def run_scene(scene, seed, outdir, n_iters=1500):
    consumers, clone_p, split_p, prune_p, n_init = _synthetic_trace(
        scene, seed, n_iters)
    records = []
    for t in sorted(consumers.keys()):
        records.append(consumer_trace.IterationRecord(
            iteration=t, n_total_rows=n_init + t,
            nonzero_grad_rows=consumers[t],
            clone_parents=clone_p.get(t, []),
            split_parents=split_p.get(t, []),
            prune_ids=prune_p.get(t, []),
            fence_rows=[t] if t % 3000 == 0 else []))
    trace_summary = consumer_trace.summarize_trace(records)
    stats = debt_replay.replay_debt(
        consumers_by_iter=consumers, n_iters=n_iters, n_initial_rows=n_init,
        clone_parents=clone_p, split_parents=split_p, prune_ids=prune_p)

    sh_wall_fraction = 0.14
    sh = headroom_gates.SceneHeadroom(scene, stats,
                                      sh_adam_wall_fraction=sh_wall_fraction)
    comparisons = _run_synthetic_exactness(scene, seed)
    cf = stats.avoidable_fraction()
    perf = perf_oracle.ScenePerf(
        scene=scene, consumer_fraction=max(0.2, 1.0 - cf),
        sh_adam_wall_fraction=sh_wall_fraction,
        t_pre_wall=1e7, t_post_wall=8e6,
        dense_sh_adam_ms_per_iter=12.0, lazy_sh_adam_ms_per_iter=6.0,
        forced_settle_overhead=0.3, microbench_speedup=None)
    return {"scene": scene, "trace_summary": trace_summary,
            "debt_stats": stats.as_dict(), "headroom": sh.as_dict(),
            "exactness": comparisons, "perf": perf.as_dict()}


def _stats_from_dict(d):
    st = debt_replay.DebtStats()
    for k, v in d.items():
        if k in st.__dict__:
            setattr(st, k, v)
    return st


def _perf_from_dict(d):
    fields = {f.name for f in dataclasses.fields(perf_oracle.ScenePerf)}
    p = perf_oracle.ScenePerf(
        scene=d.get("scene", "?"),
        consumer_fraction=d.get("consumer_fraction", 0.5),
        sh_adam_wall_fraction=d.get("sh_adam_wall_fraction", 0.1))
    for k, v in d.items():
        if k in fields:
            setattr(p, k, v)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "a100_out", "synthetic"))
    ap.add_argument("--iters", type=int, default=1500,
                    help="synthetic trace length (plumbing test, not science)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Phase A: source identity + event order
    src = source_identity.phase_a_source_gate()
    ev = event_order.phase_a_gate(max_iterations=args.iters)

    env = {
        "synthetic": True,
        "note": "SYNTHETIC pipeline run - NOT a scientific verdict. Numbers are "
                "placeholders pending the EPIC-05 A100 run.",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
    }

    per_scene = {}
    scene_results = {}
    for scene in config.SCENES:
        r = run_scene(scene, config.SEED + abs(hash(scene)) % 1000,
                      args.outdir, n_iters=args.iters)
        scene_results[scene] = r
        per_scene[scene] = {**r["headroom"], **r["perf"]}

    agg = debt_replay.aggregate_scenes({
        s: _stats_from_dict(r["debt_stats"])
        for s, r in scene_results.items()})
    headroom = {
        "passed": all(r["headroom"]["all_pass"] for r in scene_results.values()),
        "aggregate": agg,
        "per_scene": {s: r["headroom"] for s, r in scene_results.items()},
        "synthetic": True,
    }
    perf = perf_oracle.evaluate_performance({
        s: _perf_from_dict(r["perf"]) for s, r in scene_results.items()})
    perf["synthetic"] = True
    topo = {
        "exactness_pass": all(r["exactness"]["exactness_pass"]
                              for r in scene_results.values()),
        "n_fences": sum(r["exactness"]["n_fences"] for r in scene_results.values()),
        "n_mismatches": sum(r["exactness"]["n_mismatches"]
                            for r in scene_results.values()),
        "synthetic": True,
    }
    s = summary.build_summary(
        environment=env, source_identity=src,
        event_order={"resolved": ev.resolved, "notes": ev.notes,
                     "ordering": ev.ordering},
        consumer_trace={"per_scene": {x: r["trace_summary"]
                                      for x, r in scene_results.items()},
                        "synthetic": True},
        headroom=headroom, topology_oracle=topo, perf_oracle=perf)
    s["synthetic"] = True

    art = {
        "LAVOS0_ENVIRONMENT.json": env,
        "LAVOS0_SOURCE_IDENTITY.json": src,
        "LAVOS0_EVENT_ORDER.json": {"resolved": ev.resolved,
                                    "notes": ev.notes, "ordering": ev.ordering},
        "LAVOS0_CONSUMER_TRACE.json": {
            "per_scene": {x: r["trace_summary"]
                          for x, r in scene_results.items()},
            "synthetic": True},
        "LAVOS0_HEADROOM.json": headroom,
        "LAVOS0_TOPOLOGY_ORACLE.json": topo,
        "LAVOS0_PERF_ORACLE.json": perf,
        "LAVOS0_SUMMARY.json": s,
    }
    for name, obj in art.items():
        write_json(os.path.join(args.outdir, name), obj)
    write_json(
        os.path.join(args.outdir, "LAVOS0_PER_SCENE.json"),
        {x: {"debt_stats": r["debt_stats"], "exactness": r["exactness"],
              "perf": r["perf"]} for x, r in scene_results.items()})

    from lavos0 import verdict as _verdict
    v = _verdict.fold_verdict(s)
    write_json(os.path.join(args.outdir, "LAVOS0_VERDICT.json"), v)

    sys.path.insert(0, HERE)
    import lavos0_make_manifest as mm
    lines = mm.build_manifest(args.outdir)
    ok, problems = mm.verify_manifest(args.outdir)

    print(json.dumps({
        "synthetic": True,
        "verdict": v["verdict"],
        "source_identity_pass": s["source_identity_pass"],
        "event_semantics_resolved": s["event_semantics_resolved"],
        "structural_headroom_pass": s["structural_headroom_pass"],
        "topology_exactness_pass": s["topology_exactness_pass"],
        "performance_headroom_pass": s["performance_headroom_pass"],
        "manifest_ok": ok,
        "manifest_problems": problems,
        "artifacts": sorted(art.keys()) + [
            "LAVOS0_PER_SCENE.json", "LAVOS0_VERDICT.json", "MANIFEST.sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
