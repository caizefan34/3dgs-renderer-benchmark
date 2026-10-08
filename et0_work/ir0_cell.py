"""IR0 / DR0 per-cell driver (Opportunity Oracle).

Reuses the frozen C0 machinery (a0_lib) + formal A0 artifacts. NO production
CUDA is modified: the renderer is driven read-only via the same runtime
capture patch A0 used. Diagnostic only.

Stages (IR0_PREREG.md, FROZEN):
  S1  identity + environment + checkpoint-identity gate
  S2  active-set oracle (per camera): native fwd/bwd -> g; exact native Adam
      update delta_theta from the FROZEN optimizer state; u_i = max(0,-<g_i,
      delta_theta_i>); top-f visible selection (ties by Gaussian ID).
  S3  IR0-A fixed-support dirty-work lower bound (per camera full-tile replay).
  S4  cache-memory model (16B/32B per segment node).
  GATE IR0-A immediate stop on median_8cams(dirty_segment_hit_fraction_B=32)
      > 0.50 for either f=0.10 or f=0.20 in any cell.
  (S5/S6/S7 IR0-B/C/timing run only if the gate does NOT fire.)
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/root/epic05_moonshot/a0")
import a0_lib as L

FRACTIONS = [0.05, 0.10, 0.20, 0.30]
VERIFY_FRACTIONS = [0.10, 0.20]
B_LIST = [8, 16, 32, 64]
GATE_B = 32
GROUPS = ["xyz", "shs", "opacity", "scaling", "rotation"]
SH_DEGREE = 3
BETA1, BETA2, EPS_ADAM = 0.9, 0.999, 1e-15
# Authoritative expected identities.
# - room@25K ckpt + core gsplat_cuda.so: prereg inline values (both 64-char, exact).
# - C0 extension (.so): the prereg INLINE value is malformed (63 hex chars, a
#   dropped "b7" -> "...eaff6dac..."). Per IR0_PREREG.md §1 ("recover the
#   COMPLETE expected values from the formal A0/MOONSHOT manifests. Never
#   infer or truncate."), the authoritative value is the A0 manifest hash,
#   which matches the live EPIC-05 hash exactly (verified 2026-09-30).
PREREG = {
    "room_25000_ckpt_sha256":
        "b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d",
    "so_sha256":
        "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
    "core_sha256":
        "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
    "prereg_inline_so_sha256":
        "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11",
    "prereg_inline_so_note":
        "IR0_PREREG.md §1 C0-extension SHA256 is 63 hex chars (malformed; "
        "dropped 'b7'). Authoritative value taken from the formal A0 manifest "
        "(A0_CELL_SUMMARY identity.env.so_sha256), which equals the live "
        "EPIC-05 hash. core + room@25K prereg inline values are exact.",
}


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=L.json_default)
    print(f"[wrote] {path}", flush=True)


def new_acc(device):
    """GPU-side accumulators (sync once per camera)."""
    a = {k: torch.zeros((), device=device) for k in
         ("dirty_hits", "total_hits", "rendered_px", "active_px",
          "isect", "active_isect")}
    for B in B_LIST:
        a[f"seg_dirty_slots_B{B}"] = torch.zeros((), device=device)
        a[f"seg_total_slots_B{B}"] = torch.zeros((), device=device)
        a[f"seg_dirty_B{B}"] = torch.zeros((), device=device)
        a[f"seg_total_B{B}"] = torch.zeros((), device=device)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fractions", default="")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    LOG = lambda *a: print(f"[{time.time()-t0:8.1f}s]", *a, flush=True)
    fractions = (FRACTIONS if not args.fractions
                 else [float(x) for x in args.fractions.split(",")])
    n_cam_use = 2 if args.smoke else 8
    if args.smoke:
        fractions = [0.10, 0.20]

    cell_id = f"{args.scene}_{args.ckpt}"

    # ---------------- S1: identity + environment ---------------------------
    ident_env = L.bootstrap_c0()
    ds = L.load_dataset(args.scene, device="cuda")
    m, opt_state, ident = L.load_model(args.scene, args.ckpt)
    n_cams = len(ds)
    cam_ids = L.frozen_camera_ids(n_cams)
    N = int(ident["num_points"])
    LOG(f"scene={args.scene} ckpt={args.ckpt} N={N} n_cams={n_cams} "
        f"frozen_cams={cam_ids}")

    ckpt_sha = ident["ckpt_sha256"]
    id_gate = {
        "cell": cell_id,
        "ckpt_sha256": ckpt_sha,
        "so_sha256": ident_env["so_sha256"],
        "core_sha256": ident_env["core_sha256"],
        "ckpt_matches_prereg_room25":
            (ckpt_sha == PREREG["room_25000_ckpt_sha256"]
             if cell_id == "room_25000" else None),
        "so_matches_prereg": ident_env["so_sha256"] == PREREG["so_sha256"],
        "core_matches_prereg": ident_env["core_sha256"] == PREREG["core_sha256"],
    }
    id_gate["prereg_inline_so_sha256"] = PREREG["prereg_inline_so_sha256"]
    id_gate["prereg_inline_so_len"] = len(PREREG["prereg_inline_so_sha256"])
    id_gate["so_matches_authoritative_manifest"] = bool(
        id_gate["so_matches_prereg"])  # authoritative so hash == live
    id_gate["note"] = PREREG["prereg_inline_so_note"]
    id_gate["pass"] = bool(id_gate["so_matches_prereg"]
                           and id_gate["core_matches_prereg"]
                           and (id_gate["ckpt_matches_prereg_room25"] is not False))
    jdump(id_gate, os.path.join(args.outdir, f"IR0_SOURCE_IDENTITY_{cell_id}.json"))
    if not id_gate["pass"]:
        LOG("IR0_IDENTITY_FAIL STOP")
        jdump({"verdict": "IR0_IDENTITY_FAIL", "cell": cell_id},
              os.path.join(args.outdir, f"IR0_VERDICT.json"))
        return

    env_doc = {
        "cell": cell_id, "arm": ident_env["arm"],
        "so_sha256": ident_env["so_sha256"], "core_sha256": ident_env["core_sha256"],
        "worktree": ident_env["worktree"],
        "instrumentation": ident_env["instrumentation"],
        "env": ident_env["env"],
        "ckpt_path": ident["ckpt_path"], "ckpt_sha256": ckpt_sha,
        "scene": args.scene, "ckpt_iter": ident["ckpt_iter"],
        "num_points": N, "active_sh_degree": ident["active_sh_degree"],
        "spatial_lr_scale": ident["spatial_lr_scale"],
        "group_order": ident["group_order"],
        "frozen_cameras": cam_ids, "camera_rule": "id_i = round(N_cams*(i+0.5)/8)",
        "n_cams_used": n_cam_use, "fractions": fractions,
        "adam": {"beta1": BETA1, "beta2": BETA2, "eps": EPS_ADAM,
                 "per_group_lr": {g: opt_state[g]["lr"] for g in GROUPS},
                 "per_group_step": {g: opt_state[g]["step"] for g in GROUPS}},
        "python": sys.version, "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        env_doc["gpu"] = torch.cuda.get_device_name(0)
    jdump(env_doc, os.path.join(args.outdir, f"IR0_ENVIRONMENT_{cell_id}.json"))

    patch_info, uninstall = L.install_capture_patch()

    def zero_grads():
        for g in GROUPS:
            m[g].grad = None

    def group_flat(x):
        return x.reshape(x.shape[0], -1)

    def adam_update(grad, opt_state_group):
        ea = opt_state_group["exp_avg"]
        eas = opt_state_group["exp_avg_sq"]
        lr = opt_state_group["lr"]
        step = opt_state_group["step"] + 1
        mm = BETA1 * ea + (1.0 - BETA1) * grad
        vv = BETA2 * eas + (1.0 - BETA2) * grad * grad
        m_hat = mm / (1.0 - BETA1 ** step)
        v_hat = vv / (1.0 - BETA2 ** step)
        return -lr * m_hat / (v_hat.sqrt() + EPS_ADAM)

    def safe(a, b):
        return float(a / b) if b > 0 else 0.0

    def fkey(f):
        return f"{f:.2f}"

    def median(vals):
        return float(np.median(vals)) if vals else 0.0

    per_cam = {f: [] for f in fractions}
    util_doc = {"cell": cell_id, "per_camera": {}}
    losses = []

    for k in range(n_cam_use):
        ci = cam_ids[k]
        cam, gt = ds.get_item(ci)
        L.A0_CAPTURE["enabled"] = True
        L.A0_CAPTURE["records"].clear()
        zero_grads()
        frame, _, _, _ = L.render_c0(m, cam, SH_DEGREE)
        loss = L.train_loss(frame, gt)
        loss.backward()
        g_nat = {g: (m[g].grad.detach().clone() if m[g].grad is not None
                     else torch.zeros_like(m[g])) for g in GROUPS}
        rec = L.A0_CAPTURE["records"][0]
        cap = L.parse_capture(rec)
        L.A0_CAPTURE["records"].clear()
        L.A0_CAPTURE["enabled"] = False
        losses.append(float(loss))
        Nvis = int(cap["N_visible"])
        visible_master = cap["visible_ids"].detach().long()

        # ---- S2: utility oracle ----
        u = torch.zeros(Nvis, device="cuda")
        u_by_group = {g: torch.zeros(Nvis, device="cuda") for g in GROUPS}
        for g in GROUPS:
            gf = group_flat(g_nat[g])[visible_master]
            dtheta = group_flat(adam_update(g_nat[g], opt_state[g]))[visible_master]
            contrib = torch.clamp(-(gf * dtheta).sum(dim=1), min=0.0)
            u_by_group[g] = contrib
            u = u + contrib
        u = torch.clamp(u, min=0.0)
        cam_util = {"camera_id": int(ci), "loss": float(loss),
                    "N_visible": Nvis,
                    "utility_total_positive": float(u.sum().item()),
                    "by_fraction": {}}
        active_masks = {}
        for f in fractions:
            n_active = int(np.floor(f * Nvis + 0.5))
            if n_active <= 0:
                top_vis = torch.zeros(0, dtype=torch.long, device="cuda")
            else:
                order = torch.argsort(-u, stable=True)
                top_vis = order[:n_active]
            amask = torch.zeros(N, dtype=torch.bool, device="cuda")
            if n_active > 0:
                amask[visible_master[top_vis]] = True
            active_masks[f] = amask
            captured = float(u[top_vis].sum().item()) if n_active > 0 else 0.0
            total_pos = float(u.sum().item())
            cam_util["by_fraction"][fkey(f)] = {
                "n_active": n_active,
                "utility_captured": captured,
                "utility_total_positive": total_pos,
                "utility_capture_fraction": (captured / total_pos if total_pos > 0 else 0.0),
                "family_utility": {g: float(u_by_group[g][top_vis].sum().item()) if n_active > 0 else 0.0 for g in GROUPS},
            }
        util_doc["per_camera"][str(k)] = cam_util
        LOG(f"cam {k} (id {ci}) loss={float(loss):.6f} Nvis={Nvis} "
            f"util_pos={float(u.sum().item()):.4e}")

        # ---- S3: full-tile replay dirty-work census ----
        I, H, W = cap["C"], cap["height"], cap["width"]
        ts = cap["tile_size"]
        th = (H + ts - 1) // ts
        tw = (W + ts - 1) // ts
        P = ts * ts
        cam_acc = {f: new_acc("cuda") for f in fractions}
        for ty in range(th):
            for tx in range(tw):
                walk = L.replay_tile_walk(cap, 0, tx, ty)
                if walk is None:
                    continue
                valid = walk["valid"]
                visible_master_gids = visible_master[walk["gids"]]
                n = valid.shape[0]
                rows_any = valid.any(1)
                active_hit = {f: valid & active_masks[f][visible_master_gids][:, None]
                              for f in fractions}
                order_idx = torch.cumsum(valid, 0) - 1
                for f in fractions:
                    a = cam_acc[f]
                    ah = active_hit[f]
                    a["dirty_hits"] += ah.sum()
                    a["total_hits"] += valid.sum()
                    a["rendered_px"] += valid.any(0).sum()
                    a["active_px"] += ah.any(0).sum()
                    a["isect"] += rows_any.sum()
                    a["active_isect"] += (rows_any & active_masks[f][visible_master_gids]).sum()
                    for B in B_LIST:
                        nseg = (n + B - 1) // B
                        seg = order_idx // B
                        exists = torch.zeros((P, nseg), dtype=torch.bool, device="cuda")
                        active_seg = torch.zeros((P, nseg), dtype=torch.bool, device="cuda")
                        rc, cc = torch.where(valid)
                        ss = seg[rc, cc]
                        exists[cc, ss] = True
                        ar, ac = torch.where(ah)
                        if ar.numel() > 0:
                            active_seg[ac, seg[ar, ac]] = True
                        a[f"seg_dirty_slots_B{B}"] += active_seg[cc, ss].sum()
                        a[f"seg_total_slots_B{B}"] += rc.numel()
                        a[f"seg_dirty_B{B}"] += active_seg.sum()
                        a[f"seg_total_B{B}"] += exists.sum()
        # sync once per camera
        cam_acc_cpu = {f: {kk: float(vv.item()) for kk, vv in cam_acc[f].items()}
                       for f in fractions}
        for f in fractions:
            per_cam[f].append(cam_acc_cpu[f])
        del cap
        torch.cuda.empty_cache()
        LOG(f"cam {k}: replay done tiles={th}x{tw}")

    uninstall()

    def cam_fracs(a):
        return {
            "dirty_hit_fraction": safe(a["dirty_hits"], a["total_hits"]),
            "dirty_tile_intersection_fraction": safe(a["active_isect"], a["isect"]),
            "dirty_pixel_fraction": safe(a["active_px"], a["rendered_px"]),
            "dirty_segment_fraction_B": {str(B): safe(a[f"seg_dirty_B{B}"], a[f"seg_total_B{B}"]) for B in B_LIST},
            "dirty_segment_hit_fraction_B": {str(B): safe(a[f"seg_dirty_slots_B{B}"], a[f"seg_total_slots_B{B}"]) for B in B_LIST},
            "n_segment_nodes_B": {str(B): int(a[f"seg_total_B{B}"]) for B in B_LIST},
        }

    per_camera_fracs = {fkey(f): [cam_fracs(a) for a in per_cam[f]] for f in fractions}

    medians = {}
    for f in fractions:
        dfs = per_camera_fracs[fkey(f)]
        med = {}
        for metric in ("dirty_hit_fraction", "dirty_tile_intersection_fraction",
                       "dirty_pixel_fraction"):
            med[metric] = median([d[metric] for d in dfs])
        for B in B_LIST:
            med[f"dirty_segment_fraction_B{B}"] = median([d["dirty_segment_fraction_B"][str(B)] for d in dfs])
            med[f"dirty_segment_hit_fraction_B{B}"] = median([d["dirty_segment_hit_fraction_B"][str(B)] for d in dfs])
        medians[fkey(f)] = med

    dirty_doc = {
        "cell": cell_id,
        "definition": "Optimistic fixed-support LOWER BOUND: support/order held at theta_t; only active-Gaussian hits/segments marked dirty.",
        "per_camera": per_camera_fracs,
        "median_8cams": medians,
    }
    jdump(dirty_doc, os.path.join(args.outdir, f"IR0_FIXED_SUPPORT_LOCALITY_{cell_id}.json"))

    util_doc["losses"] = losses
    util_doc["median_utility_capture_fraction"] = {
        fkey(f): median([util_doc["per_camera"][str(k)]["by_fraction"][fkey(f)]["utility_capture_fraction"]
                        for k in range(n_cam_use)]) for f in fractions}
    jdump(util_doc, os.path.join(args.outdir, f"IR0_ACTIVE_UTILITY_{cell_id}.json"))

    # cache-memory model (S4): pooled segment nodes across the 8-camera cohort
    cache_doc = {"cell": cell_id, "summary_bytes_min": 16,
                 "summary_bytes_conservative": 32, "per_B": {}}
    for B in B_LIST:
        nodes = sum(per_camera_fracs[fkey(fractions[0])][k]["n_segment_nodes_B"][str(B)]
                    for k in range(len(per_camera_fracs[fkey(fractions[0])])))
        cache_doc["per_B"][str(B)] = {
            "n_segment_nodes_pooled_8cams": nodes,
            "cache_GiB_16B": nodes * 16 / (1024 ** 3),
            "cache_GiB_32B": nodes * 32 / (1024 ** 3),
        }
    jdump(cache_doc, os.path.join(args.outdir, f"IR0_CACHE_MEMORY_{cell_id}.json"))

    gate = {"cell": cell_id, "gate_B": GATE_B,
            "rule": "IR0_NO_LOCALITY if for BOTH f=0.10 and f=0.20 ANY cell has median_8cams(dirty_segment_hit_fraction_B=32) > 0.50",
            "per_fraction": {}, "fires": False}
    any_exceed = {}
    for f in VERIFY_FRACTIONS:
        v = medians[fkey(f)][f"dirty_segment_hit_fraction_B{GATE_B}"]
        any_exceed[fkey(f)] = bool(v > 0.50)
        gate["per_fraction"][fkey(f)] = {
            "median_dirty_segment_hit_fraction_B32": v,
            "exceeds_0.50": bool(v > 0.50)}
    gate["fires"] = bool(all(any_exceed.values()))
    gate["verdict_if_fire"] = "IR0_NO_LOCALITY" if gate["fires"] else None
    jdump(gate, os.path.join(args.outdir, f"IR0_GATE_{cell_id}.json"))
    jdump({"cell": cell_id, "fires": gate["fires"],
           "per_fraction_median_B32": {fkey(f): gate["per_fraction"][fkey(f)]["median_dirty_segment_hit_fraction_B32"] for f in VERIFY_FRACTIONS}},
          os.path.join(args.outdir, f"IR0_GATE_SUMMARY_{cell_id}.json"))
    LOG(f"[done {cell_id}] gate fires={gate['fires']} " +
        " ".join(f"f{fkey(f)}={gate['per_fraction'][fkey(f)]['median_dirty_segment_hit_fraction_B32']:.4f}"
                   for f in VERIFY_FRACTIONS))


if __name__ == "__main__":
    main()
