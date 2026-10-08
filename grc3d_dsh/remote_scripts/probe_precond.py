#!/usr/bin/env python3
"""M1 addendum: per-block kappa under the Adam second-moment metric.

For each bundle, re-runs the probe's per-block computation (same seed, same
block selection) and adds, per parameter group, the gradient energy measured
in the Adam second-moment metric diag(1/(sqrt(v)+eps))^2, where v is the
exp_avg_sq of the optimizer state stored in the CURRENT-iteration snapshot.

The preregistered decision uses the plain L2 kappa; this metric is the
additional "Adam-preconditioned kappa" diagnostic. Notes:
  - the plan text says "historical exp_avg_sq"; v is taken from the CURRENT
    iteration snapshot because (a) the bundle's current params come from that
    exact snapshot, so the topology/indices match exactly, and (b) evaluating
    the anchored difference at the anchor state is a mathematical identity
    (P(theta_old) = A), so the anchor-state version carries no information.
    The anchor-state identity is verified numerically and recorded.
  - optimizer state is detached and never used for updates; eps and betas are
    recorded as found. This is not a claim about Adam convergence.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gx_common as G  # noqa: E402

GROUPS = ["means", "quats", "log_scales", "opacity_logits", "sh0", "shN"]
SNAP_GROUP = {"means": "xyz", "quats": "rotation", "log_scales": "scaling",
              "opacity_logits": "opacity", "sh0": "shs", "shN": "shs"}


def group_view(t, group):
    if group == "sh0":
        return t[:, :1].reshape(t.shape[0], -1)
    if group == "shN":
        return t[:, 1:].reshape(t.shape[0], -1)
    return t.reshape(t.shape[0], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--snapshot", required=True, help="snapshot of the CURRENT iteration")
    ap.add_argument("--coarse-div", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--max-blocks", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t, ident = G.boot_gmain(args.gpu)
    G.add_grc3d_path()
    from grc3d.core import Anchor, blocks, block_losses
    from grc3d.probe import load_bundle, gradient
    from grc3d.renderers import render_gsplat

    p, old, camera, target, degree, meta = load_bundle(args.bundle, "cuda")
    h, w = target.shape[:2]
    ch, cw = max(1, h // args.coarse_div), max(1, w // args.coarse_div)

    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    osd = snap["optimizer_state_dict"]
    eps = float(osd["param_groups"][0]["eps"])
    betas = list(osd["param_groups"][0]["betas"])
    idx_name = {}
    for pg in osd["param_groups"]:
        idx_name[pg["params"][0]] = pg["name"]
    v_by_group = {}
    v_stats = {}
    for idx, name in idx_name.items():
        st = osd["state"].get(idx)
        if st is None:
            continue
        v = st["exp_avg_sq"].detach().to("cuda").float()
        step = int(st["step"]) if not torch.is_tensor(st["step"]) else int(st["step"].item())
        stats = {"step": step, "n_elements": int(v.numel()),
                 "n_v_eq_0": int((v == 0).sum()),
                 "v_min": float(v.min()), "v_max": float(v.max())}
        v_stats[name] = stats
        for g in GROUPS:
            if SNAP_GROUP[g] == name:
                vv = group_view(v, g) if name == "shs" else v.reshape(v.shape[0], -1)
                v_by_group[g] = vv
    del osd, snap
    OUT = {"bundle": args.bundle, "bundle_sha256": G.sha256_file(args.bundle),
           "snapshot": args.snapshot, "snapshot_sha256": G.sha256_file(args.snapshot),
           "v_source": {"eps": eps, "betas": betas, "per_param_group": v_stats,
                        "state_used": "CURRENT-iteration snapshot exp_avg_sq, detached"},
           "num_gaussians": len(p["means"]), "sh_degree": degree,
           "seed": args.seed, "max_blocks": args.max_blocks,
           "blocks_total": None, "per_block": [], "totals": {}}

    with torch.no_grad():
        a = Anchor.capture(render_gsplat(old, camera, h, w, degree),
                           render_gsplat(old, camera, ch, cw, degree))
        # anchor-state identity: P(theta_old) == A, delta at the cache point
        p_at_old = a.proxy(render_gsplat(old, camera, ch, cw, degree))
        OUT["anchor_identity"] = {"proxyminusA_maxabs": float((p_at_old - a.fine).abs().max()),
                                  "note": "P(theta_old) == A by construction; numeric check"}
        del p_at_old
    partition = blocks(h, w, args.block_size)
    indices = list(range(len(partition)))
    if args.max_blocks and args.max_blocks < len(indices):
        indices = torch.randperm(len(partition),
                                 generator=torch.Generator().manual_seed(args.seed)
                                 )[:args.max_blocks].tolist()
    OUT["blocks_total"] = len(partition)

    def precond_energy(x, v):
        xf = x.reshape(x.shape[0], -1).double()
        vf = v.double()
        inv = 1.0 / (vf.sqrt() + eps)
        return float((xf * inv).square().sum())

    for idx in indices:
        b = [partition[idx]]
        gf = gradient(block_losses(render_gsplat(p, camera, h, w, degree), target, b).sum(), p)
        gp = gradient(block_losses(a.proxy(render_gsplat(p, camera, ch, cw, degree)),
                                   target, b).sum(), p)
        row = {"index": idx, "center": vars(partition[idx]), "groups": {}}
        for g in GROUPS:
            d = gf[g] - gp[g]
            row["groups"][g] = {
                "fine_energy": gf[g].double().square().sum().item(),
                "difference_energy": d.double().square().sum().item(),
                "precond_fine_energy": precond_energy(gf[g], v_by_group[g]),
                "precond_difference_energy": precond_energy(d, v_by_group[g]),
            }
        del gf, gp
        OUT["per_block"].append(row)
        torch.cuda.empty_cache()

    for g in GROUPS:
        tot = {m: sum(r["groups"][g][m] for r in OUT["per_block"])
               for m in ("fine_energy", "difference_energy",
                         "precond_fine_energy", "precond_difference_energy")}
        tot["n_blocks"] = len(OUT["per_block"])
        tot["n_blocks_fine_energy_lt_1e-30"] = sum(
            1 for r in OUT["per_block"] if r["groups"][g]["fine_energy"] < 1e-30)
        tot["n_blocks_precond_fine_lt_1e-30"] = sum(
            1 for r in OUT["per_block"]
            if r["groups"][g]["precond_fine_energy"] < 1e-30)
        tot["kappa"] = (tot["difference_energy"] / tot["fine_energy"]
                        if tot["fine_energy"] >= 1e-30 else None)
        tot["precond_kappa"] = (tot["precond_difference_energy"] / tot["precond_fine_energy"]
                                if tot["precond_fine_energy"] >= 1e-30 else None)
        OUT["totals"][g] = tot
    G.write_json(args.out, OUT)
    for g in GROUPS:
        print(f"[precond] {g}: kappa={OUT['totals'][g]['kappa']} "
              f"precond_kappa={OUT['totals'][g]['precond_kappa']}", flush=True)
    print(f"[precond] anchor identity maxabs={OUT['anchor_identity']['proxyminusA_maxabs']}",
          flush=True)
    print(f"[precond] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
