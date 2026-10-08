"""VOS Phase-5 — Exact Replay Oracle.  (v2: gather/scatter write-back, name-keyed
ckpt mapping, prefix-sum trace seeking)

Arms:
  A_f : literal torch.optim.Adam with the trainer's exact constructor settings and
        DEFAULT execution path (whatever torch 2.9.1 picks for CUDA fp32 — recorded).
  A_s : torch.optim.Adam with foreach=False, fused=False (single-tensor reference).
  B   : VOS — per-row (theta, m, v, tau); deferred zero-grad transitions replayed
        step-by-step with the exact logical-step scalars before consumption.
        xyz/rotation/scaling stay DENSE in B (Phase-1 scope: only shs+opacity are
        virtualized) via B's own Adam over those tensors (bitwise control).

All arms receive IDENTICAL per-iteration gradient tensors: rows in the visible set
C_t get a deterministic pseudo-random nonzero gradient; rows outside C_t get +0.0
(exactly as the C0 backward produces: full-N zeros_like + scatter).

Bitwise note: B's row-group ops are gather -> elementwise torch ops (same kernels
and scalars as A_s) -> scatter. Elementwise ops give per-element bits independent
of tensor size/position, and gather/scatter copies are exact, so B vs A_s bitwise
identity is the expected outcome; the oracle measures it.
"""
import argparse
import json
import os
import struct
import time

import numpy as np
import torch


# ---------------------------------------------------------------- trace loading
def load_trace_segment(trace_dir, it_lo, it_hi, row_subset=None):
    """Return (list of [N] bool arrays for iters it_lo..it_hi, N, sh_degree, N_full).
    Uses meta['records'] prefix sums to seek — never loads the whole blob."""
    meta = json.load(open(os.path.join(trace_dir, "vos_trace_meta.json")))
    recs = meta["records"]  # records[i] = [iter=i+1, N, nbytes, sh_degree]
    Ns = {r[1] for r in recs[it_lo - 1:it_hi]}
    if len(Ns) != 1:
        raise RuntimeError(f"non-constant N in segment: {sorted(Ns)}")
    N_full = Ns.pop()
    # prefix offsets
    offs = {}
    off = 0
    for r in recs:
        offs[r[0]] = off
        off += 16 + r[2]
    out = []
    with open(os.path.join(trace_dir, "vos_trace.bin"), "rb") as f:
        for it in range(it_lo, it_hi + 1):
            f.seek(offs[it])
            it2, N2, nb2, shd = struct.unpack("<4i", f.read(16))
            assert it2 == it and N2 == N_full, (it2, it, N2, N_full)
            arr = np.frombuffer(f.read(nb2), dtype=np.uint8)
            bits = np.unpackbits(arr, bitorder="big")[:N_full].astype(bool)
            out.append(bits if row_subset is None else bits[row_subset])
    sh_degree = recs[it_lo - 1][3]
    return out, (N_full if row_subset is None else len(row_subset)), sh_degree, N_full


# ------------------------------------------------------- torch Adam op mirror
def torch_adam_default_flags(params):
    opt = torch.optim.Adam(params, lr=0.0025, eps=1e-15)
    g = opt.param_groups[0]
    return {"foreach": g.get("foreach"), "fused": g.get("fused"),
            "capturable": g.get("capturable"), "differentiable": g.get("differentiable")}


def vos_transition_rows(st, idx, lr, beta1, beta2, eps, step_s, grad_sub=None):
    """Apply ONE dense-Adam transition at logical step `step_s` to rows `idx`,
    mirroring torch 2.9.1 _single_tensor_adam (non-capturable, non-amsgrad, wd=0):
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias corrections + step_size on host (double)
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        param.addcdiv_(exp_avg, denom, value=-step_size)
    grad_sub=None -> grad is +0.0 (deferred transition replay).
    Gather -> op -> scatter write-back (advanced indexing copies; write-back exact).
    """
    th = st["theta"][idx]
    m = st["m"][idx]
    v = st["v"][idx]
    if grad_sub is None:
        z = torch.zeros_like(m)
        m.lerp_(z, 1 - beta1)
        v.mul_(beta2)
        v.addcmul_(z, z, value=1 - beta2)
    else:
        m.lerp_(grad_sub, 1 - beta1)
        v.mul_(beta2)
        v.addcmul_(grad_sub, grad_sub, value=1 - beta2)
    bias_correction1 = 1 - beta1 ** step_s
    bias_correction2 = 1 - beta2 ** step_s
    step_size = lr / bias_correction1
    bias_correction2_sqrt = bias_correction2 ** 0.5
    denom = (v.sqrt() / bias_correction2_sqrt).add_(eps)
    th.addcdiv_(m, denom, value=-step_size)
    st["theta"][idx] = th
    st["m"][idx] = m
    st["v"][idx] = v


def materialize(st, tau, need_mask, logical_t, lr, beta1, beta2, eps, upto_inclusive):
    """Replay deferred transitions for rows in need_mask, steps tau+1 .. upto."""
    rows_by_tau = {}
    tvals = tau[need_mask]
    uniq = torch.unique(tvals)
    for tau_val in uniq.tolist():
        rows = need_mask & (tau == tau_val)
        idx = rows.nonzero(as_tuple=False).squeeze(1)
        for s in range(tau_val + 1, upto_inclusive + 1):
            vos_transition_rows(st, idx, lr, beta1, beta2, eps, s)
        tau[idx] = upto_inclusive
        rows_by_tau[tau_val] = int(idx.numel())
    return rows_by_tau


def compare(a, b):
    af, bf = a.flatten().double(), b.flatten().double()
    diff = (af - bf).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    denom = af.abs().clamp_min(1e-30)
    max_rel = float((diff / denom).max()) if diff.numel() else 0.0
    rel_l2 = float(diff.norm() / af.norm().clamp_min(1e-30)) if af.numel() else 0.0
    if af.numel() and af.norm() > 0 and bf.norm() > 0:
        cos = float(torch.dot(af, bf) / (af.norm() * bf.norm()))
    else:
        cos = 1.0
    bitwise = bool(torch.equal(a, b))
    return {"max_abs": max_abs, "max_rel": max_rel, "rel_l2": rel_l2,
            "cosine": cos, "bitwise": bitwise}


# ------------------------------------------------------------------- the oracle
def run_oracle(cfg):
    dev = "cuda"
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    seg, N, sh_deg, N_full = load_trace_segment(
        cfg["trace_dir"], cfg["it_lo"], cfg["it_hi"], cfg.get("row_subset"))
    T = len(seg)
    vis_counts = [int(b.sum()) for b in seg]
    print(f"[oracle] segment iters {cfg['it_lo']}..{cfg['it_hi']} T={T} N={N} "
          f"(full {N_full}) sh_degree={sh_deg} "
          f"vis frac mean={np.mean(vis_counts)/N:.4f}")

    beta1, beta2 = cfg["betas"]
    eps = cfg["eps"]
    lrs = cfg["lrs"]
    shapes = {"xyz": (N, 3), "rotation": (N, 4), "scaling": (N, 3),
              "opacity": (N,), "shs": (N, 16, 3)}
    attrs = ("xyz", "shs", "opacity", "scaling", "rotation")

    # ---- initial state -----------------------------------------------------
    init = {}
    if cfg["init_mode"] == "ckpt":
        ck = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
        osd = ck["optimizer_state_dict"]
        state = osd["state"]
        by_name = {}
        for g in osd["param_groups"]:
            nm = g.get("name")
            if nm:
                by_name[nm] = state[g["params"][0]]
        for a in attrs:
            so = by_name[a]
            init[a] = {
                "theta": ck[a].float().clone(),
                "m": so["exp_avg"].float().clone(),
                "v": so["exp_avg_sq"].float().clone(),
            }
        opt_step0 = int(by_name["xyz"]["step"].item())
        print(f"[oracle] ckpt init: optimizer step counter = {opt_step0}, "
              f"N_ckpt={init['xyz']['theta'].shape[0]}")
        assert init["xyz"]["theta"].shape[0] == N_full, (
            f"ckpt N {init['xyz']['theta'].shape[0]} != trace N {N_full} — use "
            f"the matching trace/ckpt pair")
    else:
        # deterministic per-attr seeds (never hash(): string hashing is
        # process-randomized via PYTHONHASHSEED)
        attr_seed = {"xyz": 11, "shs": 22, "opacity": 33, "scaling": 44,
                     "rotation": 55}
        for a, sh in shapes.items():
            g = torch.Generator().manual_seed(cfg["seed"] + attr_seed[a])
            init[a] = {
                "theta": torch.randn(*sh, generator=g) * cfg["theta_scale"],
                "m": torch.randn(*sh, generator=g) * cfg["m_scale"],
                "v": torch.rand(*sh, generator=g) * cfg["v_scale"] + 1e-6,
            }
        opt_step0 = 0

    # row-subset the ckpt init too (ck tensors are full-N; the oracle runs on
    # the subset population so every arm must see the same rows). Random mode
    # already builds subset-sized init tensors from `shapes`.
    if cfg.get("row_subset") is not None and cfg["init_mode"] == "ckpt":
        rs = torch.from_numpy(np.ascontiguousarray(cfg["row_subset"])).long()
        for a in attrs:
            for k in ("theta", "m", "v"):
                init[a][k] = init[a][k][rs].clone()

    # synthetic opacity-reset edge case: reset_opacity zeroes opacity m/v for
    # ALL rows (verified trainer semantics); replay from zero moments must
    # still be exact (theta frozen until first nonzero grad).
    if cfg.get("zero_opacity_mv"):
        for k in ("m", "v"):
            init["opacity"][k].zero_()
        print("[oracle] synthetic opacity reset: opacity m/v zeroed")

    # ---- arm A (dense) ------------------------------------------------------
    def make_adam(foreach=None, fused=None):
        params = [init[a]["theta"].to(dev).clone().requires_grad_(True) for a in attrs]
        groups = [{"params": [p], "lr": lrs[a], "name": a}
                  for a, p in zip(attrs, params)]
        kw = {"lr": 0.0025, "eps": eps}
        if foreach is not None:
            kw["foreach"] = foreach
        if fused is not None:
            kw["fused"] = fused
        opt = torch.optim.Adam(groups, **kw)
        return opt, params

    optA_f, paramsA_f = make_adam()
    optA_s, paramsA_s = make_adam(foreach=False, fused=False)
    flags = torch_adam_default_flags([torch.zeros(4, device=dev, requires_grad=True)])
    print(f"[oracle] torch default Adam execution path: {flags}")

    stA = {}
    for opt, params in ((optA_f, paramsA_f), (optA_s, paramsA_s)):
        for p in params:
            p.grad = torch.zeros_like(p)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for a, p in zip(attrs, params):
            so = opt.state[p]
            with torch.no_grad():
                so["exp_avg"].copy_(init[a]["m"].to(dev))
                so["exp_avg_sq"].copy_(init[a]["v"].to(dev))
                so["step"].fill_(opt_step0)
        stA[id(opt)] = {a: {"theta": p, "m": opt.state[p]["exp_avg"],
                            "v": opt.state[p]["exp_avg_sq"]}
                        for a, p in zip(attrs, params)}

    # ---- arm B --------------------------------------------------------------
    dense_attrs = ("xyz", "scaling", "rotation")
    dense_params = [init[a]["theta"].to(dev).clone().requires_grad_(True)
                    for a in dense_attrs]
    optB = torch.optim.Adam(
        [{"params": [p], "lr": lrs[a], "name": a}
         for a, p in zip(dense_attrs, dense_params)], eps=eps)
    for p in dense_params:
        p.grad = torch.zeros_like(p)
    optB.step()
    optB.zero_grad(set_to_none=True)
    stB_dense = {}
    for a, p in zip(dense_attrs, dense_params):
        so = optB.state[p]
        with torch.no_grad():
            so["exp_avg"].copy_(init[a]["m"].to(dev))
            so["exp_avg_sq"].copy_(init[a]["v"].to(dev))
            so["step"].fill_(opt_step0)
        stB_dense[a] = {"theta": p, "m": so["exp_avg"], "v": so["exp_avg_sq"]}

    stB_virt = {}
    for a in ("shs", "opacity"):
        stB_virt[a] = {
            "theta": init[a]["theta"].to(dev).clone(),
            "m": init[a]["m"].to(dev).clone(),
            "v": init[a]["v"].to(dev).clone(),
            "tau": torch.full((N,), opt_step0, dtype=torch.long, device=dev),
        }

    # ---- deterministic identical gradients ----------------------------------
    ggen = torch.Generator(device=dev).manual_seed(cfg["seed"] + 12345)
    grad_scales = {"xyz": 1e-3, "shs": 1e-2, "opacity": 1e-2,
                   "scaling": 1e-3, "rotation": 1e-3}

    def gen_grads(vis_np):
        vis = torch.from_numpy(vis_np).to(dev)
        g = {}
        for a in attrs:
            sh = shapes[a]
            t = torch.zeros(sh, device=dev)
            nz = torch.randn(sh, device=dev, generator=ggen) * grad_scales[a]
            t[vis] = nz[vis]
            g[a] = t
        return g, vis

    obs_points = set(cfg["obs_points"])
    fence_points = set(cfg.get("fence_points", []))
    results = {"config": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in cfg.items()},
               "obs": [], "flags": flags, "opt_step0": opt_step0,
               "vis_counts": vis_counts, "T": T, "N": N, "N_full": N_full,
               "fence_points": sorted(fence_points)}
    replay_stats = {"materialize_events": 0, "rows_replayed": 0,
                    "transitions_replayed": 0}

    t_start = time.time()
    for t in range(1, T + 1):
        logical_t = opt_step0 + t
        grads, vis = gen_grads(seg[t - 1])

        # ---- A arms: dense step ----
        for opt, params in ((optA_f, paramsA_f), (optA_s, paramsA_s)):
            for p, a in zip(params, attrs):
                p.grad = grads[a]
            opt.step()
            opt.zero_grad(set_to_none=True)

        # ---- B: materialize consumers of t (replay tau+1 .. logical_t-1) ------
        for a in ("shs", "opacity"):
            st = stB_virt[a]
            stale = vis & (st["tau"] < logical_t - 1)
            if bool(stale.any()):
                taus = st["tau"]
                uniq = torch.unique(taus[stale])
                for tau_val in uniq.tolist():
                    rows = stale & (taus == tau_val)
                    idx = rows.nonzero(as_tuple=False).squeeze(1)
                    for s in range(tau_val + 1, logical_t):
                        vos_transition_rows(st, idx, lrs[a], beta1, beta2, eps, s)
                    st["tau"][idx] = logical_t - 1
                    replay_stats["materialize_events"] += 1
                    replay_stats["rows_replayed"] += int(idx.numel())
                    replay_stats["transitions_replayed"] += int(idx.numel()) * (logical_t - 1 - tau_val)

        # ---- B: real step for visible rows (virtualized attrs) ----
        idx_v = vis.nonzero(as_tuple=False).squeeze(1)
        for a in ("shs", "opacity"):
            st = stB_virt[a]
            vos_transition_rows(st, idx_v, lrs[a], beta1, beta2, eps, logical_t,
                                grad_sub=grads[a][idx_v])
            st["tau"][idx_v] = logical_t

        # ---- B dense part ----
        for p, a in zip(dense_params, dense_attrs):
            p.grad = grads[a]
        optB.step()
        optB.zero_grad(set_to_none=True)

        # ---- fence: full materialization (documented: placed after step t in
        #      the oracle; semantically a no-op — only exercises the path) ------
        if t in fence_points:
            for a in ("shs", "opacity"):
                st = stB_virt[a]
                need = st["tau"] < logical_t
                if bool(need.any()):
                    taus = st["tau"]
                    uniq = torch.unique(taus[need])
                    for tau_val in uniq.tolist():
                        rows = need & (taus == tau_val)
                        idx = rows.nonzero(as_tuple=False).squeeze(1)
                        for s in range(tau_val + 1, logical_t + 1):
                            vos_transition_rows(st, idx, lrs[a], beta1, beta2, eps, s)
                        st["tau"][idx] = logical_t
                        replay_stats["transitions_replayed"] += int(idx.numel()) * (logical_t + 1 - 1 - tau_val)

        # ---- observation ----
        if t in obs_points:
            for a in ("shs", "opacity"):
                st = stB_virt[a]
                need = st["tau"] < logical_t
                if bool(need.any()):
                    taus = st["tau"]
                    uniq = torch.unique(taus[need])
                    for tau_val in uniq.tolist():
                        rows = need & (taus == tau_val)
                        idx = rows.nonzero(as_tuple=False).squeeze(1)
                        for s in range(tau_val + 1, logical_t + 1):
                            vos_transition_rows(st, idx, lrs[a], beta1, beta2, eps, s)
                        st["tau"][idx] = logical_t
            entry = {"t": t, "logical_step": logical_t, "attrs": {}}
            for a in ("shs", "opacity"):
                entry["attrs"][a] = {
                    "vs_A_s": {tn: compare(stA[id(optA_s)][a][tn], stB_virt[a][tn])
                               for tn in ("theta", "m", "v")},
                    "vs_A_f": {tn: compare(stA[id(optA_f)][a][tn], stB_virt[a][tn])
                               for tn in ("theta", "m", "v")},
                }
            entry["dense_control"] = {
                a: {tn: compare(stA[id(optA_f)][a][tn], stB_dense[a][tn])
                    for tn in ("theta", "m", "v")}
                for a in dense_attrs}
            entry["A_f_vs_A_s"] = {
                a: {tn: compare(stA[id(optA_f)][a][tn], stA[id(optA_s)][a][tn])
                    for tn in ("theta", "m", "v")}
                for a in ("shs", "opacity")}
            results["obs"].append(entry)
            print(f"[oracle] obs t={t}: shs.theta vs A_s bitwise="
                  f"{entry['attrs']['shs']['vs_A_s']['theta']['bitwise']} "
                  f"max_abs={entry['attrs']['shs']['vs_A_s']['theta']['max_abs']:.3e}")

    results["wall_s"] = time.time() - t_start
    results["replay_stats"] = replay_stats

    def worst(sel):
        out = None
        for e in results["obs"]:
            for a in ("shs", "opacity"):
                for tn in ("theta", "m", "v"):
                    c = e["attrs"][a][sel][tn]
                    if out is None or c["max_abs"] > out["max_abs"]:
                        out = dict(c, attr=a, tensor=tn, t=e["t"])
        return out

    results["worst_vs_A_s"] = worst("vs_A_s")
    results["worst_vs_A_f"] = worst("vs_A_f")
    dc = [c for e in results["obs"] for a in e["dense_control"].values()
          for c in a.values()]
    results["dense_control_all_bitwise"] = all(c["bitwise"] for c in dc)
    env = [c for e in results["obs"] for a in e["A_f_vs_A_s"].values() for c in a.values()]
    results["A_f_vs_A_s_all_bitwise"] = all(c["bitwise"] for c in env)
    results["A_f_vs_A_s_worst_max_abs"] = max((c["max_abs"] for c in env), default=0.0)
    w = results["worst_vs_A_s"]
    fp32_pass = (w is not None and w["max_abs"] <= cfg["fp32_max_abs"] and
                 w["rel_l2"] <= cfg["fp32_rel_l2"] and w["cosine"] >= cfg["fp32_cos"])
    results["verdict"] = {
        "fp32_equivalent_vs_A_s": bool(fp32_pass),
        "bitwise_vs_A_s": bool(w is not None and w["bitwise"]),
        "bitwise_vs_A_f": bool(results["worst_vs_A_f"] is not None and
                               results["worst_vs_A_f"]["bitwise"]),
        "dense_control_bitwise": results["dense_control_all_bitwise"],
        "note": ("bitwise_vs_A_f False with A_f_vs_A_s nonzero = foreach-vs-single "
                 "kernel arithmetic envelope, not a VOS defect"),
    }
    print(json.dumps(results["verdict"], indent=2))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--it-lo", type=int, required=True)
    ap.add_argument("--it-hi", type=int, required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--init-mode", choices=["random", "ckpt"], default="random")
    ap.add_argument("--n-rows", type=int, default=100000,
                    help="row subset size (0 = all rows)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--obs-points", type=str, default="50,137,299,500,777,1000")
    ap.add_argument("--fence-points", type=str, default="400")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--zero-opacity-mv", action="store_true",
                    help="zero opacity m/v after init (synthetic opacity-reset edge case)")
    args = ap.parse_args()

    cfg = {
        "trace_dir": args.trace_dir,
        "it_lo": args.it_lo, "it_hi": args.it_hi,
        "ckpt": args.ckpt, "init_mode": args.init_mode,
        "seed": args.seed,
        "betas": [0.9, 0.999],
        "eps": 1e-15,
        "lrs": {"xyz": 0.00016 * 0.5, "shs": 0.0025, "opacity": 0.025,
                "scaling": 0.005, "rotation": 0.001},
        "obs_points": [int(x) for x in args.obs_points.split(",") if x],
        "fence_points": [int(x) for x in args.fence_points.split(",") if x],
        "fp32_max_abs": 1e-5, "fp32_rel_l2": 1e-6, "fp32_cos": 0.9999999,
        "theta_scale": 0.1, "m_scale": 1e-3, "v_scale": 1e-5,
        "zero_opacity_mv": bool(args.zero_opacity_mv),
    }
    if args.n_rows > 0:
        meta = json.load(open(os.path.join(args.trace_dir, "vos_trace_meta.json")))
        N_full = meta["records"][args.it_lo - 1][1]
        rs = np.random.RandomState(args.seed)
        cfg["row_subset"] = np.sort(rs.choice(N_full, size=min(args.n_rows, N_full),
                                              replace=False))
    else:
        cfg["row_subset"] = None

    os.makedirs(args.outdir, exist_ok=True)
    res = run_oracle(cfg)
    out = os.path.join(args.outdir, f"oracle_{args.tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
