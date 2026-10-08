#!/usr/bin/env python3
"""PGR0 Projection Gradient Redundancy Oracle (P0, CPU-only, no CUDA).

For the frozen gsplat fused 3DGS projection path (EPIC-05 config: packed=False,
pinhole, eps2d=0.3, near=0.01, far=1e10), this oracle measures whether the
projection BACKWARD performs redundant computation relative to the FORWARD:

  R1 projected-mean redundancy : backward-recomputed mean2d vs forward mean2d
  R2 covariance redundancy      : backward-recomputed covar2d vs forward covar2d
  R3 Jacobian redundancy        : backward-recomputed J      vs forward J
  R4 gradient redundancy        : reuse-path gradient vs autograd ground truth
                                  (and recompute-path gradient vs autograd)

Each record is one (scene, camera, gaussian-population) cell. The oracle is
fail-closed: it never claims redundancy below threshold and never runs CUDA.
"""
from __future__ import annotations
import argparse, json, math, os, sys, hashlib, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import pgr0_projection_ref as P

# ---- frozen EPIC-05 config -------------------------------------------------
CFG = dict(packed=False, tile_size=16, accutile=False, radius_clip=0.0,
           eps2d=0.3, near_plane=0.01, far_plane=1e10, camera_model="pinhole",
           gsplat_expected="1.5.3")

# ---- frozen gates ----------------------------------------------------------
GATES = dict(
    sim_threshold=0.999,        # cosine/relative similarity to accept redundancy
    rel_l2_threshold=1e-4,      # max allowed relative L2 error (fp32 emulation)
    min_redundancy_rate=0.90,   # min fraction of active gaussians with redundancy
    min_optimistic_speedup=1.05,# Amdahl-style upper bound to keep the candidate
)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def set_scene(scene, n, seed):
    """Deterministic, scene-flavored Gaussian population + cameras (no external
    data needed; projection math is per-gaussian and scene-agnostic)."""
    g = torch.Generator().manual_seed(seed)
    # scene flavor: spread/scale/rotation jitter
    flavor = {
        "room":    dict(span=3.0, sc=0.06, sj=0.05, qj=0.15, zlo=1.5, zhi=6.0),
        "bicycle": dict(span=4.0, sc=0.05, sj=0.08, qj=0.30, zlo=1.0, zhi=8.0),
        "garden":  dict(span=5.0, sc=0.07, sj=0.06, qj=0.20, zlo=1.2, zhi=9.0),
    }[scene]
    means = (torch.rand(n, 3, generator=g) - 0.5) * flavor["span"]
    means[:, 2] = flavor["zlo"] + torch.rand(n, generator=g) * (flavor["zhi"] - flavor["zlo"])
    quats = torch.randn(n, 4, generator=g) * flavor["qj"]
    quats = torch.nn.functional.normalize(quats, dim=-1)
    quats[:, 0] += 1.0  # near-identity rotation
    quats = torch.nn.functional.normalize(quats, dim=-1)
    scales = torch.rand(n, 3, generator=g) * flavor["sj"] + flavor["sc"] * 0.5
    # two cameras
    C = 2
    vm = torch.zeros(C, 4, 4)
    for ci in range(C):
        vm[ci] = torch.eye(4)
        vm[ci, 2, 3] = -(3.0 + 0.5 * ci)
        vm[ci, 0, 3] = 0.2 * ci
    W, H = 960, 540
    Ks = torch.tensor([[[600., 0, W / 2], [0, 600., H / 2], [0, 0, 1]]] * C, dtype=torch.float64)
    return means, quats, scales, vm, Ks, W, H


def cos_sim(a, b, eps=1e-30):
    a = a.reshape(-1).double(); b = b.reshape(-1).double()
    c = float((a @ b) / (a.norm() * b.norm() + eps))
    return max(-1.0, min(1.0, c))  # clamp against float rounding


def rel_l2(a, b, eps=1e-30):
    a = a.reshape(-1).double(); b = b.reshape(-1).double()
    return float((a - b).norm() / (b.norm() + eps))


def run_cell(scene, cam, n, seed, dtype, calc_compensations):
    torch.manual_seed(seed)
    dev = "cpu"
    means, quats, scales, vm, Ks, W, H = set_scene(scene, n, seed)
    means = means.to(dtype); quats = quats.to(dtype); scales = scales.to(dtype)
    vm = vm.to(dtype); Ks = Ks.to(dtype)
    inp = P.ProjInputs(means.clone(), quats.clone(), scales.clone(), vm.clone(), Ks, W, H,
                       eps2d=CFG["eps2d"], near_plane=CFG["near_plane"], far_plane=CFG["far_plane"])
    fwd = P.forward(inp, calc_compensations=calc_compensations)
    active = (fwd.radii[cam, :, 0] > 0) & (fwd.radii[cam, :, 1] > 0)  # (N,)
    n_active = int(active.sum().item())
    if n_active == 0:
        return None
    # recompute path intermediates for this camera
    ri = P.recompute_intermediates(inp)
    idx = active.nonzero(as_tuple=True)[0]
    # R1 projected mean
    f_mean = fwd.means2d[cam][idx]
    r_mean = ri["mean2d"][cam][idx]
    # R2 covariance
    f_cov = fwd.covar2d_preblur[cam][idx]
    r_cov = ri["covar2d"][cam][idx]
    # R3 Jacobian
    f_J = fwd.J[cam][idx]
    r_J = ri["J"][cam][idx]
    # gradient seeds (simulate rasterizer adjoint), per (camera, gaussian)
    g = torch.Generator().manual_seed(seed + 977)
    Cn = vm.shape[0]
    v_m2d = torch.randn(Cn, n, 2, generator=g, dtype=torch.float64).to(dtype)
    v_dep = torch.randn(Cn, n, 1, generator=g, dtype=torch.float64).to(dtype)
    v_con = torch.randn(Cn, n, 3, generator=g, dtype=torch.float64).to(dtype)
    v_cmp = torch.randn(Cn, n, 1, generator=g, dtype=torch.float64).to(dtype) if calc_compensations else None
    # mask to active gaussians
    am = active.view(1, n, 1)
    v_m2d = v_m2d * am; v_dep = v_dep * am; v_con = v_con * am
    if v_cmp is not None:
        v_cmp = v_cmp * am
    # backward() consumes the full (C,n,...) batch; compare full gradients.
    vdep_full = v_dep[..., 0]
    vcmp_full = v_cmp[..., 0] if v_cmp is not None else None
    ag = P.autograd_grad(inp, v_m2d, vdep_full, v_con, vcmp_full, calc_compensations=calc_compensations)
    rc = P.backward(inp, fwd, v_m2d, vdep_full, v_con, vcmp_full, mode="recompute", calc_compensations=calc_compensations)
    ru = P.backward(inp, fwd, v_m2d, vdep_full, v_con, vcmp_full, mode="reuse", calc_compensations=calc_compensations)
    # R4 gradient similarity (reuse vs autograd, recompute vs autograd)
    grad_keys = ("v_means", "v_quats", "v_scales", "v_viewmats")
    grad_sim = {}
    for k in grad_keys:
        grad_sim[k] = {
            "reuse_vs_autograd_cos": cos_sim(ru[k], ag[k]),
            "reuse_vs_autograd_rel": rel_l2(ru[k], ag[k]),
            "recompute_vs_autograd_cos": cos_sim(rc[k], ag[k]),
            "recompute_vs_autograd_rel": rel_l2(rc[k], ag[k]),
        }
    rec = {
        "scene": scene, "camera": cam, "n": n, "n_active": n_active,
        "dtype": str(dtype).split(".")[-1].replace(")", ""),
        "calc_compensations": calc_compensations,
        "seed": seed,
        "R1_projected_mean": {"cos": cos_sim(f_mean, r_mean), "rel_l2": rel_l2(f_mean, r_mean)},
        "R2_covariance": {"cos": cos_sim(f_cov, r_cov), "rel_l2": rel_l2(f_cov, r_cov)},
        "R3_jacobian": {"cos": cos_sim(f_J, r_J), "rel_l2": rel_l2(f_J, r_J)},
        "R4_gradient": grad_sim,
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "..", "artifacts", "records", "pgr0_records.jsonl"))
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--compensations", default="both", choices=["both", "on", "off"])
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--scenes", default="room,bicycle,garden")
    args = ap.parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    comp_modes = {"both": [False, True], "on": [True], "off": [False]}[args.compensations]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seed0 = 20261001
    rows = []
    with open(args.out, "w", encoding="utf-8") as f:
        ci = 0
        for scene in args.scenes.split(","):
            for cam in (0, 1):
                for comp in comp_modes:
                    seed = seed0 + ci
                    rec = run_cell(scene, cam, args.n, seed, dtype, comp)
                    ci += 1
                    if rec is None:
                        continue
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                    rows.append(rec)
                    print("CELL %s cam=%d comp=%s n_active=%d "
                          "R1=%.6f R2=%.6f R3=%.6f R4_reuse_quat_cos=%.6f" % (
                              scene, cam, rec["calc_compensations"], rec["n_active"],
                              rec["R1_projected_mean"]["cos"], rec["R2_covariance"]["cos"],
                              rec["R3_jacobian"]["cos"],
                              rec["R4_gradient"]["v_quats"]["reuse_vs_autograd_cos"]), flush=True)
    print("WROTE %d records -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
