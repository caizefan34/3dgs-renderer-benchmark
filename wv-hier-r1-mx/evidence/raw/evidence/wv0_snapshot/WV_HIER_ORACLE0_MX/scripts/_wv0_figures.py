#!/usr/bin/env python3
"""WV-HIER-ORACLE0-MX figures.
Reads WV0_GRADIENT_SUMMARY.pt + work, produces PNG figures per spec section 28:
- gradient NMSE vs structural work (per scene, per budget)
- 1-cosine vs structural work
- granularity vs gradient error
- granularity vs work
Separate room/garden/bicycle; identify early/mid/late, p=0.25/0.40.
"""
import sys, os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/wv0_out"
FIGDIR = os.path.join(OUT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

agg = torch.load(os.path.join(OUT, "WV0_GRADIENT_SUMMARY.pt"), weights_only=False)
grad = agg["grad"]; work = agg["work"]; meta = agg["states"]
GRAN = [16, 8, 4, 1]
BUDGETS = [0.25, 0.40]
SCENE_COLOR = {"room": "tab:blue", "garden": "tab:green", "bicycle": "tab:red"}
STAGE_MARK = {"5000": "o", "15000": "s", "30000": "^"}

def stage_of(sid):
    for k in ("5000","15000","30000"):
        if f"iter{k}" in sid: return k
    return "other"

def ckey(G,p): return f"G{G}_p{p}"

# structural work W_int for a state at budget p, granularity G:
# selected isects / dense isects = p (uniform). But the MEANINGFUL work axis for the
# Pareto plot is the per-state dense isect count at G (the structural work a correct
# backend would process if it selected that granularity's tiles at budget p).
# We plot x = dense_isects_G * p  (selected structural work) normalized by dense isects_G16 * 1.0
# Better: x = W_int(G) = (selected isects at G) / (dense isects at G16 full).
def W_int(sid, G, p):
    G = str(G)
    sel = work[sid]["per_G"][G]["dense_isects"] * p   # selected isects at granularity G
    dense16 = work[sid]["per_G"]["16"]["dense_isects"]  # full-image isects at G16
    return sel / dense16

# ---- Figure 1: granularity vs gradient error (geometry NMSE macro), per scene, per budget ----
for p in BUDGETS:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, sc in zip(axes, ["room","garden","bicycle"]):
        sids = [s for s in grad if meta[s]["scene"] == sc]
        for sid in sids:
            xs = GRAN; ys = [grad[sid][ckey(G,p)]["geom_nmse_mean"] for G in GRAN]
            mk = STAGE_MARK.get(stage_of(sid), "x")
            ax.plot(xs, ys, marker=mk, color=SCENE_COLOR[sc], alpha=0.8, label=stage_of(sid))
        ax.set_xscale("log", base=2); ax.set_xticks(GRAN); ax.set_xticklabels([str(g) for g in GRAN])
        ax.set_xlabel("granularity G (px tile)"); ax.set_ylabel("geometry NMSE (macro)")
        ax.set_title(f"{sc}  p={p}"); ax.grid(True, alpha=0.3)
        ax.axhline(0.70, color="gray", ls="--", lw=0.8, alpha=0.5)  # Gate A threshold ref
    axes[0].legend(title="stage")
    fig.suptitle(f"Granularity vs geometry-gradient NMSE  (p={p})", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, f"fig_gran_vs_nmse_p{p}.png"), dpi=110); plt.close(fig)

# ---- Figure 2: granularity vs work (selected structural work W_int vs G16 dense) ----
for p in BUDGETS:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, sc in zip(axes, ["room","garden","bicycle"]):
        sids = [s for s in grad if meta[s]["scene"] == sc]
        for sid in sids:
            xs = GRAN; ys = [W_int(sid, G, p) for G in GRAN]
            mk = STAGE_MARK.get(stage_of(sid), "x")
            ax.plot(xs, ys, marker=mk, color=SCENE_COLOR[sc], alpha=0.8, label=stage_of(sid))
        ax.set_xscale("log", base=2); ax.set_xticks(GRAN); ax.set_xticklabels([str(g) for g in GRAN])
        ax.set_xlabel("granularity G"); ax.set_ylabel("W_int (selected isects / dense-G16 isects)")
        ax.set_title(f"{sc}  p={p}"); ax.grid(True, alpha=0.3)
    axes[0].legend(title="stage")
    fig.suptitle(f"Granularity vs structural work  (p={p})", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, f"fig_gran_vs_work_p{p}.png"), dpi=110); plt.close(fig)

# ---- Figure 3 & 4: NMSE vs work and 1-cos vs work (Pareto plane), per scene per budget ----
for p in BUDGETS:
    for metric, fname, ylab in [("nmse", "fig_nmse_vs_work", "geometry NMSE (macro)"),
                                 ("cos", "fig_1mcos_vs_work", "1 - cosine")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, sc in zip(axes, ["room","garden","bicycle"]):
            sids = [s for s in grad if meta[s]["scene"] == sc]
            for sid in sids:
                mk = STAGE_MARK.get(stage_of(sid), "x")
                for G in GRAN:
                    x = W_int(sid, G, p)
                    if metric == "nmse":
                        y = grad[sid][ckey(G,p)]["geom_nmse_mean"]
                    else:
                        y = 1.0 - grad[sid][ckey(G,p)]["geom_cos_mean"]
                    ax.scatter([x], [y], marker=mk, color=SCENE_COLOR[sc], s=40, alpha=0.8)
                # connect the granularity path
                xs = [W_int(sid, G, p) for G in GRAN]
                ys = [grad[sid][ckey(G,p)]["geom_nmse_mean"] if metric=="nmse" else 1.0-grad[sid][ckey(G,p)]["geom_cos_mean"] for G in GRAN]
                ax.plot(xs, ys, color=SCENE_COLOR[sc], alpha=0.3, lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("structural work W_int (vs dense-G16)"); ax.set_ylabel(ylab)
            ax.set_title(f"{sc}  p={p}"); ax.grid(True, alpha=0.3)
        fig.suptitle(f"{ylab} vs structural work  (p={p})", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, f"{fname}_p{p}.png"), dpi=110); plt.close(fig)

print("figures written to", FIGDIR)
for f in sorted(os.listdir(FIGDIR)):
    print(" ", f)
