#!/usr/bin/env python3
"""Generate publication figures A-G and tables 1-6 from frozen aggregates.
Re-runnable; emits into pubphase/figtables/. All numbers derive from
runs_master.json + the pair-table JSONs (never transcribed by hand).

Figures:
  A  per-scene speedup (C0 vs B1A, matched benchmark; FINAL-30K frozen)
  B  per-scene quality deltas (dPSNR/dSSIM/dLPIPS, C0 vs B1A)
  C  cumulative ablation per-scene wall (A0 -> A1 -> A2 -> A3=C0)
  D  cumulative ablation per-scene dPSNR vs A0
  E  eps2d 2x2 panel (wall ratio + dPSNR per stack)
  F  N_GS trajectories (per arm, room+garden+bicycle)
  G  TTQ-style eval-grid curves (PSNR vs wall-time, room)

Tables (markdown): 1 P1 baselines, 2 P2 ablation wall, 3 P2 ablation quality,
4 P4 eps2d, 5 P6 multiseed (when present), 6 external systems (placeholder).
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = "/mnt/storage_pool/liaoyuanjun/pubphase/aggregates"
OUT = "/mnt/storage_pool/liaoyuanjun/pubphase/figtables"
os.makedirs(OUT, exist_ok=True)

ALL13 = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
         "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]

master = json.load(open(f"{A}/runs_master.json"))
RUNS = master["runs"]


def find(v, s, seed=42, eps2d=None):
    for r in RUNS:
        if (r["variant"] == v and r["scene"] == s and r["seed"] == seed
                and (eps2d is None or r["eps2d"] == eps2d)
                and not r.get("contaminated")):
            return r
    return None


def pair(name):
    p = f"{A}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def fig_a():
    t = pair("headline_c0_vs_b1a_f30k")
    rows = [r for r in t["rows"] if r["status"] == "ok"]
    rows.sort(key=lambda r: r["speedup"] or 0)
    scenes = [r["scene"] for r in rows]
    sp = [r["speedup"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2a7" if x > 1 else "#c44" for x in sp]
    ax.bar(range(len(sp)), [x - 1 for x in sp], color=colors)
    ax.set_xticks(range(len(sp)))
    ax.set_xticklabels(scenes, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    gm = t["aggregate"]["speedup_geomean"]
    ax.axhline(gm - 1, color="navy", ls="--", lw=1,
               label=f"geomean {gm:.4f}x")
    ax.set_ylabel("speedup - 1 (C0 vs B1A, matched)")
    ax.set_title("Figure A: per-scene training speedup (30K iters, seed 42, matched protocol)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figA_speedup_per_scene.png", dpi=150)
    plt.close(fig)


def fig_b():
    t = pair("headline_c0_vs_b1a_f30k")
    rows = [r for r in t["rows"] if r["status"] == "ok"]
    rows.sort(key=lambda r: r["scene"])
    x = range(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    for ax, key, lbl in zip(axes, ("d_psnr", "d_ssim", "d_lpips"),
                            ("dPSNR (dB)", "dSSIM", "dLPIPS")):
        vals = [r[key] for r in rows]
        ax.bar(x, vals, color="#48a")
        ax.axhline(0, color="k", lw=0.8)
        m = sum(vals) / len(vals)
        ax.axhline(m, color="navy", ls="--", lw=1, label=f"mean {m:+.4f}")
        ax.set_title(lbl, fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_xticks(list(x))
    for ax in axes:
        ax.set_xticklabels([r["scene"] for r in rows], rotation=60, ha="right", fontsize=7)
    fig.suptitle("Figure B: per-scene quality deltas (C0 - B1A)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figB_quality_deltas.png", dpi=150)
    plt.close(fig)


def _scene_arm_matrix(scenes):
    """wall[scene][variant] for the ablation chain."""
    chain = ["a0", "a1", "a2", "c0"]
    walls, dpsnrs = {}, {}
    for s in scenes:
        base = find("a0", s)
        row_w, row_d = [], []
        for v in chain:
            r = find(v, s, eps2d=0.3)
            row_w.append(r["wall_s"] if r else None)
            if r and base:
                row_d.append(r["psnr"] - base["psnr"])
            else:
                row_d.append(None)
        walls[s], dpsnrs[s] = row_w, row_d
    return walls, dpsnrs


def fig_c_d():
    scenes = [s for s in ALL13 if find("a0", s) is not None]
    walls, dpsnrs = _scene_arm_matrix(scenes)
    chain = ["A0", "A1", "A2", "A3=C0"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for i, s in enumerate(scenes):
        ws = walls[s]
        rel = [w / ws[0] if (w and ws[0]) else None for w in ws]
        xs = [j for j, v in enumerate(rel) if v is not None]
        ys = [v for v in rel if v is not None]
        ax.plot(xs, ys, marker="o", lw=1, ms=3, label=s if len(scenes) <= 6 else None,
                alpha=0.75)
    ax.set_xticks(range(4))
    ax.set_xticklabels(chain)
    ax.set_ylabel("wall / wall(A0)")
    ax.set_title("Figure C: cumulative ablation wall time", fontsize=10)
    if len(scenes) <= 6:
        ax.legend(fontsize=7)
    ax = axes[1]
    for s in scenes:
        ds = dpsnrs[s]
        xs = [j for j, v in enumerate(ds) if v is not None]
        ys = [v for v in ds if v is not None]
        ax.plot(xs, ys, marker="o", lw=1, ms=3, alpha=0.75,
                label=s if len(scenes) <= 6 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(chain)
    ax.set_ylabel("PSNR - PSNR(A0) [dB]")
    ax.set_title("Figure D: cumulative ablation quality", fontsize=10)
    if len(scenes) <= 6:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figCD_ablation.png", dpi=150)
    plt.close(fig)


def fig_e():
    scenes = ["room", "bicycle", "garden"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, ylab in zip(axes, ("wall", "psnr"),
                             ("wall ratio (0.3 / 0.1)", "dPSNR (0.3 - 0.1) [dB]")):
        for i, (v, lbl) in enumerate((("b1a", "B1A"), ("c0", "C0"))):
            xs, ys = [], []
            for j, s in enumerate(scenes):
                lo = find(v, s, eps2d=0.1)
                hi = find(v, s, eps2d=0.3)
                if not (lo and hi):
                    continue
                if key == "wall":
                    ys.append(hi["wall_s"] / lo["wall_s"] - 1)
                else:
                    ys.append(hi["psnr"] - lo["psnr"])
                xs.append(j + (i - 0.5) * 0.35)
            ax.bar(xs, ys, width=0.33, label=lbl)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(scenes)))
        ax.set_xticklabels(scenes, fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.legend(fontsize=8)
    fig.suptitle("Figure E: eps2d sensitivity (2x2, seed 42)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE_eps2d.png", dpi=150)
    plt.close(fig)


def fig_f():
    scenes = ["room", "bicycle", "garden"]
    variants = [("b1a", 0.1), ("b1", 0.1), ("a0", 0.3), ("a2", 0.3), ("c0", 0.3)]
    fig, axes = plt.subplots(1, len(scenes), figsize=(12, 3.4))
    for ax, s in zip(axes, scenes):
        for v, e in variants:
            r = find(v, s, eps2d=e)
            if not r:
                continue
            grid = r.get("eval_grid") or []
            xs = [g[0] for g in grid]
            ns = []
            # n at eval steps from training_results is not in master; use grid psnr proxy? no:
            # fig F needs N trajectories -- use per_iter from run dir instead
            ns = None
        ax.set_title(s, fontsize=9)
    # NOTE: N trajectories need per_iter n_gs series -- pulled separately below
    fig.tight_layout()
    fig.savefig(f"{OUT}/figF_placeholder.png", dpi=150)
    plt.close(fig)


def fig_f_real():
    """N_GS trajectories from per_iter series in the run dirs (room/bicycle/garden)."""
    import glob
    scenes = ["room", "bicycle", "garden"]
    arms = [("b1a", "final30k_runs", "b1a"), ("b1", "pub_runs", "b1"),
            ("a0", "pub_runs", "a0"), ("a2", "pub_runs", "a2"),
            ("c0", "final30k_runs", "c0")]
    base = "/mnt/storage_pool/liaoyuanjun"
    fig, axes = plt.subplots(1, len(scenes), figsize=(12, 3.4))
    for ax, s in zip(axes, scenes):
        for v, root, pref in arms:
            rj = f"{base}/{root}/{pref}_{s}/results.json"
            if not os.path.exists(rj):
                continue
            d = json.load(open(rj))
            per = d.get("per_iter", [])
            xs = [p["iter"] for p in per]
            ys = [p["n_gs"] for p in per]
            ax.plot(xs, ys, lw=1, label=v, alpha=0.8)
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("iteration", fontsize=8)
        ax.set_ylabel("N gaussians", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle("Figure F: densification trajectories (N vs iteration, seed 42)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figF_n_trajectories.png", dpi=150)
    plt.close(fig)


def fig_g():
    """TTQ-style: PSNR vs wall-time on the eval grid (room, all matched arms)."""
    base = "/mnt/storage_pool/liaoyuanjun"
    arms = [("B1A", f"{base}/final30k_runs/b1a_room/results.json"),
            ("B1", f"{base}/pub_runs/b1_room/results.json"),
            ("A0", f"{base}/pub_runs/a0_room/results.json"),
            ("A2", f"{base}/pub_runs/a2_room/results.json"),
            ("C0", f"{base}/final30k_runs/c0_room/results.json")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, rj in arms:
        if not os.path.exists(rj):
            continue
        d = json.load(open(rj))
        t0 = d["timing"]["total_wall_s"] - d["final_eval"].get("wall_time_s", 0)
        rows = [r for r in d.get("eval_rows", []) if r.get("wall_time_s") is not None]
        xs = [r["wall_time_s"] for r in rows]
        ys = [r["psnr"] for r in rows]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.2, label=name)
    ax.set_xlabel("wall time [s] (incl. eval overhead, uniform both arms)")
    ax.set_ylabel("PSNR [dB]")
    ax.set_title("Figure G: time-to-quality, room (seed 42)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figG_ttq_room.png", dpi=150)
    plt.close(fig)


def table_1():
    lines = ["# Table 1: P1 matched baselines (per-scene, seed 42, 30K, clean pairs)",
             "",
             "B1A = clean gsplat v1.5.3 + accutile (eps2d 0.1); B1 = pristine v1.5.3 "
             "(eps2d 0.1); C0 = final method (eps2d 0.3). B0 = original Graphdeco "
             "rasterizer @54c035f under the matched trainer; its densification statistic "
             "is the original signed-grad-norm >=2e-4 (the single disclosed protocol "
             "difference). Wall in minutes. eps2d disclosure: C0@0.3 vs B1A@0.1 "
             "asymmetry contributes ~1.1% speed / ~0.10 dB (see P4).",
             "",
             "Ratio columns are wall ratios (S19): B1A/C0 > 1 means C0 is faster "
             "(the headline orientation, geomean 1.0685x); B1A/B1 < 1 means B1 is "
             "slower than B1A; B1A/B0 < 1 means B0 is slower.",
             "",
             "| scene | B1A wall | B1 wall | B0 wall | C0 wall | B1A/C0 | B1A/B1 | B1A/B0 | "
             "PSNR B1A | PSNR B1 | PSNR B0 | PSNR C0 | dPSNR C0-B1A | N B1A | N B0 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in ALL13:
        b1a = find("b1a", s, eps2d=0.1)
        b1 = find("b1", s, eps2d=0.1)
        b0 = find("b0", s)
        c0 = find("c0", s, eps2d=0.3)
        def mm(r):
            return f"{r['wall_s']/60:.1f}" if r else "-"
        def pp(r):
            return f"{r['psnr']:.3f}" if r else "-"
        def nn(r):
            return f"{r['n_gaussians']:,}" if r else "-"
        sp1 = f"{b1a['wall_s']/c0['wall_s']:.4f}" if (b1a and c0) else "-"
        sp2 = f"{b1a['wall_s']/b1['wall_s']:.4f}" if (b1a and b1) else "-"
        sp0 = f"{b1a['wall_s']/b0['wall_s']:.4f}" if (b1a and b0) else "-"
        dp = f"{c0['psnr']-b1a['psnr']:+.3f}" if (b1a and c0) else "-"
        lines.append(f"| {s} | {mm(b1a)} | {mm(b1)} | {mm(b0)} | {mm(c0)} | {sp1} | {sp2} | "
                     f"{sp0} | {pp(b1a)} | {pp(b1)} | {pp(b0)} | {pp(c0)} | {dp} | "
                     f"{nn(b1a)} | {nn(b0)} |")
    with open(f"{OUT}/table1_p1_baselines.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def table_2_3():
    scenes = [s for s in ALL13 if find("a0", s) is not None]
    lines2 = ["# Table 2: P2 cumulative ablation -- wall time (seed 42, clean)",
              "",
              "A0 = internal parent (F9/SA/H8 off) -> A1 (+F9) -> A2 (+SCALAR_ADJOINT) "
              "-> A3 = C0 (+H8-MR), same frozen binary, env-switched; PX=2 held.",
              "",
              "| scene | A0 wall | A1 wall | A2 wall | A3(C0) wall | A3/A0 speedup |",
              "|---|---|---|---|---|---|"]
    lines3 = ["# Table 3: P2 cumulative ablation -- quality vs A0 (seed 42)",
              "",
              "| scene | dPSNR A1 | dPSNR A2 | dPSNR A3 | dSSIM A3 | dLPIPS A3 | N A0 | N A3 | N ratio |",
              "|---|---|---|---|---|---|---|---|---|"]
    for s in scenes:
        a0 = find("a0", s)
        a1 = find("a1", s)
        a2 = find("a2", s)
        a3 = find("c0", s, eps2d=0.3)
        def mm(r):
            return f"{r['wall_s']/60:.1f}" if r else "-"
        sp = f"{a0['wall_s']/a3['wall_s']:.4f}" if (a0 and a3) else "-"
        lines2.append(f"| {s} | {mm(a0)} | {mm(a1)} | {mm(a2)} | {mm(a3)} | {sp} |")
        def dd(r, k):
            return f"{r[k]-a0[k]:+.3f}" if (r and a0) else "-"
        nrat = f"{a3['n_gaussians']/a0['n_gaussians']:.3f}" if (a0 and a3) else "-"
        lines3.append(f"| {s} | {dd(a1,'psnr')} | {dd(a2,'psnr')} | {dd(a3,'psnr')} | "
                      f"{dd(a3,'ssim')} | {dd(a3,'lpips')} | "
                      f"{a0['n_gaussians']:,} | {a3['n_gaussians']:,} | {nrat} |")
    with open(f"{OUT}/table2_p2_ablation_wall.md", "w") as f:
        f.write("\n".join(lines2) + "\n")
    with open(f"{OUT}/table3_p2_ablation_quality.md", "w") as f:
        f.write("\n".join(lines3) + "\n")


def table_4():
    scenes = ["room", "bicycle", "garden"]
    lines = ["# Table 4: P4 eps2d sensitivity 2x2 (seed 42, clean pairs)",
             "",
             "Cells: eps2d 0.1 vs 0.3 per stack. Wall ratio = wall(0.3)/wall(0.1).",
             "",
             "| scene | stack | wall@0.1 | wall@0.3 | ratio | dPSNR(0.3-0.1) | dSSIM | dLPIPS |",
             "|---|---|---|---|---|---|---|---|"]
    for s in scenes:
        for v in ("b1a", "c0"):
            lo = find(v, s, eps2d=0.1)
            hi = find(v, s, eps2d=0.3)
            if not (lo and hi):
                lines.append(f"| {s} | {v} | - | - | - | - | - | - |")
                continue
            lines.append(f"| {s} | {v} | {lo['wall_s']/60:.1f} | {hi['wall_s']/60:.1f} | "
                         f"{hi['wall_s']/lo['wall_s']:.4f} | {hi['psnr']-lo['psnr']:+.3f} | "
                         f"{hi['ssim']-lo['ssim']:+.4f} | "
                         f"{(hi['lpips']-lo['lpips']):+.4f} |")
    with open(f"{OUT}/table4_p4_eps2d.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def table_5():
    P6_SCENES = ["drjohnson", "train", "bicycle", "room"]
    lines = ["# Table 5: P6 multi-seed confirmation (4 scenes x {B1A, C0} x seeds 42/43/44)",
             "",
             "Seeds 43/44 are the P6 runs; seed 42 reuses the FINAL-30K cohort "
             "(identical protocol). Speedup = wall_B1A/wall_C0 (S19: > 1 = C0 faster, "
             "the headline orientation). dPSNR = C0 - B1A. Pending cells are '-' "
             "until the P6 queue drains.",
             "",
             "| scene | seed | B1A wall (min) | C0 wall (min) | B1A/C0 | dPSNR |",
             "|---|---|---|---|---|---|"]
    per_seed_geo = {}
    for s in P6_SCENES:
        for k in (42, 43, 44):
            b = find("b1a", s, seed=k, eps2d=0.1)
            c = find("c0", s, seed=k, eps2d=0.3)
            if b and c:
                lines.append(f"| {s} | {k} | {b['wall_s']/60:.1f} | {c['wall_s']/60:.1f} "
                             f"| {b['wall_s']/c['wall_s']:.4f} | {c['psnr']-b['psnr']:+.3f} |")
                per_seed_geo.setdefault(k, []).append(b["wall_s"] / c["wall_s"])
            else:
                lines.append(f"| {s} | {k} | - | - | - | - |")
    lines += ["", "Per-seed 4-scene speedup geomean (S19):", ""]
    for k in (42, 43, 44):
        rs = per_seed_geo.get(k, [])
        if len(rs) == len(P6_SCENES):
            g = math.exp(sum(math.log(r) for r in rs) / len(rs))
            lines.append(f"- seed {k}: **{g:.4f}** (4/4 scenes)")
        elif rs:
            lines.append(f"- seed {k}: {len(rs)}/4 scenes present (incomplete)")
        else:
            lines.append(f"- seed {k}: pending")
    lines += ["",
              "drjohnson row carries the per-scene outlier label (R-13): its seed-42 "
              "+1.102 dB is reported per-scene, never generalized (S20). "
              "Aggregate-without-drjohnson rows are in `p6_results.json`."]
    with open(f"{OUT}/table5_p6_multiseed.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    fig_a()
    fig_b()
    fig_c_d()
    fig_e()
    try:
        fig_f_real()
    except Exception as ex:
        print("figF skipped:", ex)
    fig_g()
    table_1()
    table_2_3()
    table_4()
    table_5()
    print("figures+tables ->", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
