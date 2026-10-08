#!/usr/bin/env python3
"""UCR0 — Unbiased Control-Variate Raster Feasibility Oracle (FROZEN prereg).

Executes on the remote EPIC-05 A100 (NEVER local GPU). Reuses the frozen C0
primitives from n1_core.py. Topology frozen: ONE forward/backward/Adam step per
probe, no densify/prune. See UCR0_PREREG.md.
"""
import argparse
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
from fused_ssim import FusedSSIMMap

C1 = 0.01 ** 2
C2 = 0.03 ** 2
LAM = C.LAMBDA_DSSIM          # 0.2 frozen
TS = 16                       # loss-unit tile size (SSIM-center)
CELLS = [("room", 10000), ("room", 15000), ("bicycle", 10000), ("bicycle", 15000)]
CKPT_HASH = {
    ("room", 10000): "9aa00f0ddf28b6c097e3dc8f41dbb107cd9bf175750503dc330aa1d6ab579fc4",
    ("room", 15000): "b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd",
    ("bicycle", 10000): "26bf0b959f10f0ce526c496d0dc7ca8e62b0ba93efc83d58b7f9440d58c2f4e9",
    ("bicycle", 15000): "273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7",
}
PREREG_PIN_C0 = "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11"
PREREG_PIN_CORE = "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"
QS = ["0.05", "0.10"]
N_SOURCE = 4
N_PEER = 3
N_HOLDOUT = 8
N_MASKS = 16

# HM0 measured per-phase wall fractions (HM0_SUMMARY stageA_dynamic_profile_frac)
HM0_PHASE = {
    ("room", 10000):   dict(camera=0.06618154267640343, forward=0.22135111731791846, l1=0.012755522699447365, ssim=0.01861074996279817, backward=0.5340502503392758, optimizer=0.1382677067087548),
    ("room", 15000):   dict(camera=0.06989192462038156, forward=0.2122035943033941, l1=0.012355409508701313, ssim=0.017960844951027254, backward=0.5245705110325253, optimizer=0.15544610533866524),
    ("bicycle", 10000): dict(camera=0.03957419063091352, forward=0.18368909595629254, l1=0.007454232843505463, ssim=0.010861324384045629, backward=0.4550580138321035, optimizer=0.29926922741358225),
    ("bicycle", 15000): dict(camera=0.03407413558229238, forward=0.1808949086376059, l1=0.00672981800665412, ssim=0.009747673522733823, backward=0.43719556659742, optimizer=0.32766284541327245),
}


def _stable_seed(s):
    import hashlib
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little") % (2 ** 63)


def cost_model(cell, q):
    """HM0-based ideal cost oracle (prereg sections 5/7).

    proxy+residual cost fraction = Q4 proxy (1/16 raster work) + q residual,
    relative to full raster work. Ideal E2E speed upper bound = 1/(1/16+q).
    Also reports the honest full-iteration fraction (incl. Adam optimizer) as
    context (HM0 measured camera+optimizer are irreducible per-iteration).
    """
    qf = float(q)
    ph = HM0_PHASE[cell]
    fixed = ph["camera"] + ph["optimizer"]
    raster = ph["forward"] + ph["backward"]
    loss = ph["l1"] + ph["ssim"]
    cost_fraction = 1.0 / 16.0 + qf                 # ideal raster+loss fraction
    e2e_ideal = 1.0 / cost_fraction                 # ideal upper bound
    full_frac = fixed + (raster + loss) * cost_fraction
    e2e_full = 1.0 / full_frac
    return {"predicted_cost_fraction": cost_fraction,
            "predicted_e2e_speedup": e2e_ideal,
            "hm0_fixed_frac": fixed, "hm0_raster_frac": raster,
            "hm0_loss_frac": loss,
            "full_training_iter_frac": full_frac,
            "full_training_e2e_speedup": e2e_full}


def stratified(pool, k):
    N = len(pool); used = set(); out = []
    for i in range(k):
        p = int(math.floor((i + 0.5) * N / k))
        while p in used or p >= N:
            p += 1
        used.add(p); out.append(p)
    return out


def camera_semantics(ds):
    n = len(ds)
    train = [i for i in range(n) if i % 8 != 0]
    holdout = [i for i in range(n) if i % 8 == 0]
    assert set(train).isdisjoint(set(holdout))
    return n, train, holdout


def build_panels(train, holdout, centers):
    """4 midpoint-stratified TRAIN source cameras; 3 nearest TRAIN peers/source
    (by camera-center distance, deterministic tie-break by camera id);
    8 midpoint-stratified true-holdout cameras."""
    spos = stratified(train, N_SOURCE)
    sources = [train[p] for p in spos]
    train_set = set(train)
    peers_of = {}
    for s in sources:
        sc = centers[s]
        cand = []
        for j in train:
            if j == s:
                continue
            d = float(np.linalg.norm(sc - centers[j]))
            cand.append((d, j, s))
        cand.sort()
        peers_of[s] = [c[1] for c in cand[:N_PEER]]
    hpos = stratified(holdout, N_HOLDOUT)
    hold_panel = [holdout[p] for p in hpos]
    return sources, peers_of, hold_panel


def q4_camera(cam, H, W):
    """Quarter-linear-resolution camera for the Q4 proxy (K/4, dims/4)."""
    Kq = cam.K.clone()
    Kq[0, 0] /= 4.0; Kq[0, 2] /= 4.0; Kq[1, 1] /= 4.0; Kq[1, 2] /= 4.0
    return types.SimpleNamespace(K=Kq, viewmatrix=cam.viewmatrix,
                                 image_width=W // 4, image_height=H // 4,
                                 camera_center=cam.camera_center)


def decompose(b, g):
    """Per-pixel loss map from full-res CHW tensors b,g.
    pix_loss.mean() == frozen fused loss L = 0.8*L1 + 0.2*(1-SSIM)."""
    ssim_map = FusedSSIMMap.apply(C1, C2, b, g, "same", True, 2)  # (1,3,H,W)
    ssim_pix = ssim_map.mean(dim=1, keepdim=True)                  # (1,1,H,W)
    l1_map = (b - g).abs().mean(dim=1, keepdim=True)               # (1,1,H,W)
    return (1.0 - LAM) * l1_map + LAM * (1.0 - ssim_pix)           # (1,1,H,W)


def tile_pixel_mask(tile_tiles, H, W, device):
    """tile_tiles: (nt_r,nt_c) bool. Returns (1,1,H,W) pixel indicator."""
    m = tile_tiles.to(device).float()
    m = m.repeat_interleave(TS, 0).repeat_interleave(TS, 1)[:H, :W]
    return m.view(1, 1, H, W)


def lr_of_param(model, p):
    for pg in model.optimizer.param_groups:
        if any(p is q for q in pg["params"]):
            return pg["lr"]
    return None


def opt_unit(grads, lrs):
    return [gr.detach() / lr for gr, lr in zip(grads, lrs)]


def dot_opt(a, b):
    return sum(float((x * y).sum()) for x, y in zip(a, b))


def norm_opt(a):
    return math.sqrt(sum(float(x.pow(2).sum()) for x in a))


def cosine_opt(a, b):
    na, nb = norm_opt(a), norm_opt(b)
    return dot_opt(a, b) / (na * nb) if na > 0 and nb > 0 else float("nan")


def set_grads(model, grads):
    params = [getattr(model, n) for n in C.PARAM_NAMES]
    for p, gr in zip(params, grads):
        if p.grad is None:
            p.grad = gr.detach().clone()
        else:
            p.grad.copy_(gr.detach())


def loss_parity_gate(model, ds, source_cams, cell, log):
    """Prereg section 2 gate (run before UCR probes)."""
    params = [getattr(model, n) for n in C.PARAM_NAMES]
    lrs = [lr_of_param(model, p) for p in params]
    ci = source_cams[0]
    cam, gt = ds.get_item(ci)
    img, _, _ = C.render_view(model, cam, model.active_sh_degree)
    b = img.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    g = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    H, W = b.shape[-2], b.shape[-1]
    fused = C.make_fused_loss()
    L_frozen = fused(img, gt)
    pix_loss = decompose(b, g)
    L_map = float(pix_loss.double().sum() / (H * W))
    Lf = float(L_frozen.detach())
    scalar_rel = abs(L_map - Lf) / abs(Lf)
    gf = torch.autograd.grad(L_frozen, params, retain_graph=True, allow_unused=True)
    gnorm_opt = math.sqrt(sum(float((gr.detach() * lr).pow(2).sum())
                             for gr, lr in zip(gf, lrs) if gr is not None))
    errs = []
    for s in range(16):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(_stable_seed("UCR0PARITY|%s|%d" % (cell, s)))
        dirs = []
        for p, lr in zip(params, lrs):
            r = torch.randint(0, 2, tuple(p.shape), generator=gen,
                              device="cuda", dtype=torch.int8).float().mul_(2).sub_(1)
            dirs.append(r * lr)

        def dd(lt):
            gr = torch.autograd.grad(lt, params, retain_graph=True, allow_unused=True)
            return sum(float((gi * vi).sum()) for gi, vi in zip(gr, dirs) if gi is not None)
        Df = dd(L_frozen); Dd = dd(pix_loss.mean())
        errs.append(abs(Dd - Df) / gnorm_opt)
    errs = np.array(errs)
    med = float(np.median(errs)); mx = float(errs.max())
    passed = bool(scalar_rel <= 1e-6 and med <= 1e-4 and mx <= 5e-4)
    log("  [parity %s] scalar_rel=%.3e dir_med=%.3e dir_max=%.3e -> %s"
        % (cell, scalar_rel, med, mx, "PASS" if passed else "FAIL"))
    return {"scalar_rel_err": scalar_rel, "dir_rel_median": med, "dir_rel_max": mx,
            "loss_parity_pass": passed, "gnorm_opt": gnorm_opt}


def make_masks(cell, source_idx, q, nt_r, nt_c):
    qf = float(q)
    masks = []
    for mi in range(N_MASKS):
        seed = _stable_seed("UCRMASK|%s@%d|src%d|q%s|m%d" % (cell[0], cell[1], source_idx, q, mi))
        gen = torch.Generator(device="cuda"); gen.manual_seed(seed % (2 ** 63))
        u = torch.rand(nt_r, nt_c, generator=gen, device="cuda")
        masks.append(u < qf)
    return masks


@torch.no_grad()
def eval_state(model, ds, cams, fused):
    """cams: list of camera indices. Returns {ci: (loss, psnr)}."""
    out = {}
    for ci in cams:
        cam, gt = ds.get_item(ci)
        img, _, _ = C.render_view(model, cam, model.active_sh_degree)
        loss = float(fused(img, gt).item())
        bb = img.unsqueeze(0).permute(0, 3, 1, 2)
        gg = gt.unsqueeze(0).permute(0, 3, 1, 2)
        mse = float(F.mse_loss(bb, gg).item())
        psnr = float(20 * math.log10(1.0 / math.sqrt(mse))) if mse > 1e-10 else 100.0
        out[ci] = (loss, psnr)
    return out


def process_source_cell(model, ds, source, peers, hold_panel, cell, q, log):
    """One (cell, source, q): 16 frozen Bernoulli masks -> CV + HT estimator.
    Phase 1: ALL backwards with params frozen at theta0 (retained graphs).
    Phase 2: native Adam steps + source/peer/true-holdout evaluation."""
    qf = float(q)
    params = [getattr(model, n) for n in C.PARAM_NAMES]
    lrs = [lr_of_param(model, p) for p in params]
    master = C.StateMaster(model)
    fused = C.make_fused_loss()
    cam, gt = ds.get_item(source)
    H, W = cam.image_height, cam.image_width

    # ---- phase 1: exact gradient + proxy + per-mask residuals (theta0 frozen) ----
    img, _, _ = C.render_view(model, cam, model.active_sh_degree)
    b = img.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    g = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    pix_loss = decompose(b, g)                                  # (1,1,H,W)
    G_exact = torch.autograd.grad(pix_loss.mean(), params, retain_graph=True, allow_unused=True)
    G_exact_opt = opt_unit(G_exact, lrs)
    gnorm_exact = norm_opt(G_exact_opt)

    cam_q4 = q4_camera(cam, H, W)
    img_q4, _, _ = C.render_view(model, cam_q4, model.active_sh_degree)
    img_proxy = F.interpolate(img_q4.permute(2, 0, 1).unsqueeze(0), size=(H, W),
                              mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)
    bp = img_proxy.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    proxy_pix_loss = decompose(bp, g)                           # (1,1,H,W)
    C_proxy = torch.autograd.grad(proxy_pix_loss.mean(), params, retain_graph=True, allow_unused=True)

    nt_r = math.ceil(H / TS); nt_c = math.ceil(W / TS)
    masks = make_masks(cell, source, q, nt_r, nt_c)
    G_hat_store = []; recs = []; nan_inf = 0
    for mi, mask_tiles in enumerate(masks):
        mask_pix = tile_pixel_mask(mask_tiles, H, W, b.device)
        g_S = torch.autograd.grad((pix_loss * mask_pix).sum() / (H * W), params,
                                  retain_graph=True, allow_unused=True)
        c_S = torch.autograd.grad((proxy_pix_loss * mask_pix).sum() / (H * W), params,
                                  retain_graph=True, allow_unused=True)
        G_hat = [C_proxy[i] + (g_S[i] - c_S[i]) / qf for i in range(len(params))]
        HT_hat = [g_S[i] / qf for i in range(len(params))]
        for t in G_hat:
            if not bool(torch.isfinite(t).all()):
                nan_inf += 1
        G_hat_opt = opt_unit(G_hat, lrs)
        HT_opt = opt_unit(HT_hat, lrs)
        cos_cv = cosine_opt(G_hat_opt, G_exact_opt)
        cos_ht = cosine_opt(HT_opt, G_exact_opt)
        mse_cv = sum(float((a - bb).pow(2).sum()) for a, bb in zip(G_hat_opt, G_exact_opt))
        mse_ht = sum(float((a - bb).pow(2).sum()) for a, bb in zip(HT_opt, G_exact_opt))
        G_hat_store.append([t.detach().clone() for t in G_hat])
        recs.append({"cos_cv": cos_cv, "cos_ht": cos_ht, "mse_cv": mse_cv, "mse_ht": mse_ht,
                     "n_incl": int(mask_tiles.sum())})
        del g_S, c_S, G_hat, HT_hat, G_hat_opt, HT_opt
    del pix_loss, proxy_pix_loss, b, g, img, img_q4, img_proxy, bp, mask_pix
    torch.cuda.empty_cache()

    # ---- phase 2: native Adam steps + evaluation (params mutate here) ----
    eval_cams = [source] + list(peers) + list(hold_panel)
    master.restore()
    L0 = eval_state(model, ds, eval_cams, fused)
    master.restore(); set_grads(model, G_exact); model.optimizer.step()
    L_exact = eval_state(model, ds, eval_cams, fused)
    L0_src = L0[source][0]; L0_peer = float(np.mean([L0[p][0] for p in peers]))
    Lex_src = L_exact[source][0]; Lex_peer = float(np.mean([L_exact[p][0] for p in peers]))
    improvement_peer = L0_peer - Lex_peer
    denom_src = L0_src - Lex_src
    for mi, G_hat in enumerate(G_hat_store):
        master.restore(); set_grads(model, G_hat); model.optimizer.step()
        Lc = eval_state(model, ds, eval_cams, fused)
        Lc_src = Lc[source][0]; Lc_peer = float(np.mean([Lc[p][0] for p in peers]))
        hold_deltas = [Lc[h][1] - L_exact[h][1] for h in hold_panel]
        progress = (L0_src - Lc_src) / denom_src if abs(denom_src) > 1e-12 else float("nan")
        peer_frac = (Lc_peer - Lex_peer) / improvement_peer if abs(improvement_peer) > 1e-12 else float("nan")
        recs[mi].update({"progress": progress, "peer_frac": peer_frac, "hold_deltas": hold_deltas})
    mean_raw = [sum(Gh[i] for Gh in G_hat_store) / N_MASKS for i in range(len(params))]
    mean_opt = opt_unit(mean_raw, lrs)
    bias = norm_opt([m - e for m, e in zip(mean_opt, G_exact_opt)]) / gnorm_exact
    return {"recs": recs, "nan_inf": nan_inf, "bias_rel": bias,
            "gnorm_exact": gnorm_exact, "improvement_peer": improvement_peer,
            "n_tiles": nt_r * nt_c}


def aggregate_cell(source_results, cell, q):
    """Pool 4 sources x 16 masks -> per-(cell,q) metrics for the analyzer."""
    all_recs = []
    biases = []
    nan_inf = 0
    improvement = []
    for sr in source_results:
        all_recs.extend(sr["recs"])
        biases.append(sr["bias_rel"])
        nan_inf += sr["nan_inf"]
        improvement.append(sr["improvement_peer"])
    cos_cv = np.array([r["cos_cv"] for r in all_recs], dtype=float)
    cos_ht = np.array([r["cos_ht"] for r in all_recs], dtype=float)
    mse_cv = float(np.mean([r["mse_cv"] for r in all_recs]))
    mse_ht = float(np.mean([r["mse_ht"] for r in all_recs]))
    progress = np.array([r["progress"] for r in all_recs], dtype=float)
    peer_frac = np.array([r["peer_frac"] for r in all_recs], dtype=float)
    hold = np.array([d for r in all_recs for d in r["hold_deltas"]], dtype=float)
    var_red = (mse_ht / mse_cv) if mse_cv > 0 else float("nan")
    cm = cost_model(cell, q)
    return {"loss_parity_pass": True,   # parity is global; per-cell gate checked separately
            "bias_rel": float(np.median(biases)),
            "cos_median": float(np.median(cos_cv)),
            "cos_p10": float(np.percentile(cos_cv, 10)),
            "cos_ht_median": float(np.median(cos_ht)),
            "mse_cv": mse_cv, "mse_ht": mse_ht,
            "variance_reduction_vs_ht": float(var_red),
            "progress_capture_median": float(np.nanmedian(progress)),
            "progress_capture_p10": float(np.nanpercentile(progress, 10)),
            "peer_delta_fraction_median": float(np.nanmedian(peer_frac)),
            "truehold_psnr_delta_median_db": float(np.nanmedian(hold)),
            "truehold_psnr_delta_worst_db": float(np.nanmin(hold)),
            "predicted_cost_fraction": cm["predicted_cost_fraction"],
            "predicted_e2e_speedup": cm["predicted_e2e_speedup"],
            "nan_inf_count": int(nan_inf),
            "n_masks": len(all_recs), "n_sources": len(source_results),
            "improvement_peer_median": float(np.median(improvement))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/epic05_ucr0")
    ap.add_argument("--cells", default="room@10000,room@15000,bicycle@10000,bicycle@15000")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "ucr0_raw.log"), "w", buffering=1)
    def P(*a):
        s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")

    cells = [tuple(x.split("@")) for x in args.cells.split(",")]
    cells = [(s, int(i)) for s, i in cells]
    if args.smoke:
        cells = cells[:1]

    # ---- A. identity + environment ------------------------------------------
    ident = C.bootstrap_c0()
    env = {"host": os.uname().nodename, "gpu": "NVIDIA A100-SXM4-80GB",
           "python": sys.version.split()[0], "torch": torch.__version__,
           "cuda_available": bool(torch.cuda.is_available()), "device": "cuda:0",
           "loss_backend": "fused-ssim (padding=same) -- frozen n1snap training loss",
           "loss_so_sha256": ident["so_sha256"],
           "c0_ext_so_match_prereg_pin": ident["so_sha256"] == PREREG_PIN_C0,
           "core_so_match_prereg_pin": ident["core_sha256"] == PREREG_PIN_CORE,
           "eps2d": C.EPS2D, "lambda_dssim": C.LAMBDA_DSSIM, "tile_size": TS,
           "background": "black", "topology": "frozen during one-step probe",
           "estimator": "G_hat = C + sum_{u in S}(g_u - c_u)/q ; HT_hat = sum_{u in S} g_u/q",
           "proxy": "Q4 (W/4,H/4) render, bilinear RGB lift, full-res loss units"}
    P("[A] identity c0_ext=%s core=%s" % (ident["so_sha256"][:16], ident["core_sha256"][:16]))
    assert env["c0_ext_so_match_prereg_pin"] and env["core_so_match_prereg_pin"], "BINARY MISMATCH"
    with open(os.path.join(args.out, "UCR0_ENVIRONMENT.json"), "w") as f:
        json.dump(env, f, indent=2)

    source_identity = {"audit_type": "UCR0_SOURCE_IDENTITY",
                       "authority": "AUTHORITATIVE_ASSETS.json + OPEN_TRACK_ASSET_AUDIT_20260930",
                       "c0_ext_so_sha256": ident["so_sha256"],
                       "core_gsplat_so_sha256": ident["core_sha256"],
                       "prereg_pin_c0": PREREG_PIN_C0, "prereg_pin_core": PREREG_PIN_CORE,
                       "c0_ext_so_match": ident["so_sha256"] == PREREG_PIN_C0,
                       "core_so_match": ident["core_sha256"] == PREREG_PIN_CORE,
                       "checkpoints": {}}
    for scene, it in cells:
        ck = os.path.join(C.SNAP_RUNS[scene], f"ckpt_{it}.pt")
        source_identity["checkpoints"][f"{scene}@{it}"] = {
            "path": ck, "sha256": C.sha256_file(ck),
            "expected": CKPT_HASH[(scene, it)],
            "match": C.sha256_file(ck) == CKPT_HASH[(scene, it)]}
        assert source_identity["checkpoints"][f"{scene}@{it}"]["match"], f"CKPT MISMATCH {scene}@{it}"
    with open(os.path.join(args.out, "UCR0_SOURCE_IDENTITY.json"), "w") as f:
        json.dump(source_identity, f, indent=2)

    base = {}
    results = {}
    parity_all_pass = True
    cam_sem = {}
    for scene, it in cells:
        t0 = time.perf_counter()
        P(f"\n===== CELL {scene}@{it} =====")
        ckpt_path = os.path.join(C.SNAP_RUNS[scene], f"ckpt_{it}.pt")
        h = C.sha256_file(ckpt_path)
        assert h == CKPT_HASH[(scene, it)], f"CKPT HASH MISMATCH {scene}@{it}: {h}"
        ds = base.get(scene) or C.load_scene_dataset(scene)
        base[scene] = ds
        n, train, holdout = camera_semantics(ds)
        centers = {i: ds.get_camera(i).camera_center.detach().float().cpu().numpy() for i in train}
        sources, peers_of, hold_panel = build_panels(train, holdout, centers)
        P(f"  n={n} train={len(train)} holdout={len(holdout)} sources={sources} hold_panel={hold_panel}")
        cam_sem[f"{scene}@{it}"] = {
            "n_cams": n, "train_pool_size": len(train), "true_holdout_size": len(holdout),
            "source_cams": sources,
            "peers_of_source": {str(s): peers_of[s] for s in sources},
            "true_holdout_panel": hold_panel,
            "rule": "TRUE HOLDOUT = canonical index i % 8 == 0; TRAIN POOL = rest",
            "train_intersect_holdout_empty": bool(set(train).isdisjoint(holdout))}
        ckpt = C.load_checkpoint(scene, it)
        model = C.build_model(ckpt)
        C.scene_reset()
        # ---- C. loss decomposition parity gate (first source) ----
        par = loss_parity_gate(model, ds, sources, (scene, it), P)
        parity_all_pass = parity_all_pass and par["loss_parity_pass"]
        if not par["loss_parity_pass"]:
            P("  UCR0_LOSS_DECOMPOSITION_FAIL")
            results[f"{scene}@{it}"] = {"parity": par, "failed": True}
            continue
        cell_res = {"parity": par, "q": {}}
        for q in QS:
            src_results = []
            for si, source in enumerate(sources):
                ts = time.perf_counter()
                sr = process_source_cell(model, ds, source, peers_of[source],
                                         hold_panel, (scene, it), q, P)
                sr["source"] = source; sr["secs"] = time.perf_counter() - ts
                src_results.append(sr)
                mc = np.mean([r["n_incl"] for r in sr["recs"]])
                P(f"  q={q} src={source} bias={sr['bias_rel']:.4f} "
                  f"cos_med={np.median([r['cos_cv'] for r in sr['recs']]):.4f} "
                  f"var_red={np.mean([r['mse_ht'] for r in sr['recs']])/max(np.mean([r['mse_cv'] for r in sr['recs']]),1e-30):.2f} "
                  f"mask_incl={mc:.0f}/{sr['n_tiles']} ({sr['secs']:.1f}s)")
            agg = aggregate_cell(src_results, (scene, it), q)
            agg["per_source"] = [{"source": sr["source"], "bias_rel": sr["bias_rel"],
                                  "gnorm_exact": sr["gnorm_exact"]} for sr in src_results]
            cell_res["q"][q] = agg
        results[f"{scene}@{it}"] = cell_res
        P(f"  cell {scene}@{it} done in {time.perf_counter()-t0:.1f}s; "
          f"peak_gpu={C.gpu_mem_gb():.1f}GB")

    with open(os.path.join(args.out, "UCR0_CAMERA_SEMANTICS.json"), "w") as f:
        json.dump(cam_sem, f, indent=2)

    # ---- E. summary ----------------------------------------------------------
    summary = {"loss_decomposition_global_pass": bool(parity_all_pass), "cells": {}}
    for cell_name, cr in results.items():
        if cr.get("failed"):
            summary["cells"][cell_name] = {"failed": True, "parity": cr.get("parity")}
            continue
        summary["cells"][cell_name] = {"parity": cr["parity"], "q": cr["q"]}
    with open(os.path.join(args.out, "UCR0_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)
    P("\nwrote UCR0_SUMMARY.json")
    P("loss_decomposition_global_pass=%s" % parity_all_pass)

    # ---- F. verdict (frozen analyzer) ---------------------------------------
    analyzer = "/root/epic05_ucr0/ucr0_analyze.py"
    import shutil
    if os.path.abspath("ucr0_analyze.py") != os.path.abspath(analyzer):
        shutil.copy("ucr0_analyze.py", analyzer)
    rc = os.system(f"/root/miniforge3/envs/gsplat/bin/python {analyzer} {os.path.join(args.out,'UCR0_SUMMARY.json')}")
    P("analyzer exit=%d" % rc)
    log.close()


if __name__ == "__main__":
    main()
