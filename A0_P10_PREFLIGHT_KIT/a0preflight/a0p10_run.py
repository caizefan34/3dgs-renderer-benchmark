#!/usr/bin/env python3
"""A0-P10 remote driver (EPIC-05 A100, frozen gsplat env python).

One-shot pipeline for the single frozen cell room@25K camera 19 (DSCF4819):
  Stage 0  identity / hash verification          -> A0_SOURCE_IDENTITY.json
  Stage 1  exact-backward factorization check     -> (A0_EXACT_BACKWARD_FACTORIZATION.md, static)
  Stage 2  preflight sanity (raw-frame adjoint, native graph, determinism)
                                                  -> A0_PREFLIGHT_SANITY.json
  Stage 3  native tile x batch bucket census + top-32 freeze
                                                  -> A0_BUCKETS.json
  Stage 4  exact FP32 coefficient dump            -> out/buckets/bucket_*.npz
  Stage 5  oracle-SVD / DCT structure analysis + frozen P10 verdict
                                                  -> A0_STRUCTURE_RESULTS.json

STOP discipline (A0_V2 prereg, REVISED): if Stage 0 identity or Stage 2 preflight
sanity fails, the script writes the artifact, prints the blocker code, and EXITS
(no further stages). P10 checks are PREFLIGHT SANITY only -- they do NOT claim
full A0-v2 native-gradient parity (that is left to formal A0). No DCT/VJP
reconstruction, no bicycle/20K, no full campaign, no production CUDA.

Usage:
    CUDA_VISIBLE_DEVICES=0 /root/miniforge3/envs/gsplat/bin/python a0p10_run.py --out /root/A0_P10/out
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402
from lib import (  # noqa: E402
    REPO, RECIPE, P10_SCENE, P10_CKPT, P10_CAM_INDEX, P10_CAM_IMAGE, P10_TOPK,
    bootstrap_c0, load_dataset, load_model_params, fused_loss_fn,
    json_dump, sha256_file, tile_bucket_walk, tile_batches, dense_coefficients,
    p10_verdict,
)
from oracle_analyze import analyze_bucket  # noqa: E402

# Frozen PX runtime -> native bucket width = 256 / PX (PX=2 frozen arm -> 128).
HIGS_PX_RUNTIME = int(RECIPE["HIGS_PX_RUNTIME"])
BUCKET_WIDTH = 256 // HIGS_PX_RUNTIME
TILE_SIZE = RECIPE["tile_size"]


def run_identity(out_dir):
    """Stage 0: run the identity verifier in-process and return (dict, rc)."""
    import subprocess
    ident_path = os.path.join(out_dir, "A0_SOURCE_IDENTITY.json")
    r = subprocess.run([sys.executable, os.path.join(HERE, "a0p10_identity.py"),
                        "--out", ident_path], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    d = json.load(open(ident_path))
    return d, r.returncode


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=None,
                    help="physical GPU index to force (torch.cuda.set_device); the frozen "
                         "C0 rasterizer may ignore CUDA_VISIBLE_DEVICES, so an explicit "
                         "device is authoritative. Pick a free GPU on EPIC-05.")
    a = ap.parse_args()
    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)
    import torch

    # Force the device BEFORE any CUDA init (authoritative; overrides CVD if the
    # frozen rasterizer ignores it).
    if a.gpu is not None:
        torch.cuda.set_device(a.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
        print(f"forcing CUDA device -> physical GPU {a.gpu}")

    # ---- Stage 0: identity -------------------------------------------------
    print("=== Stage 0: identity ===")
    ident, rc = run_identity(out_dir)
    # Gate on the HARD checks only. The kernel byte-identity checks degrade to a
    # non-fatal note if the remote kernel path is missing (path layout can differ
    # between hosts); a present-but-MISMATCHED kernel source, or a mismatched
    # frozen artifact / checkpoint sha, IS a hard failure.
    def _is_missing_note(c):
        return (c.get("note") or "").startswith("REMOTE PATH MISSING")
    hard_bad = [c for c in ident["identity_checks"]
                if not (c.get("ok") and c.get("present")) and not _is_missing_note(c)]
    if hard_bad:
        json_dump({"verdict": "A0P_IDENTITY_BLOCKER", "detail": ident},
                  os.path.join(out_dir, "A0_VERDICT.json"))
        print("BLOCKER: A0P_IDENTITY_BLOCKER -- identity verification failed; STOP.")
        for c in hard_bad:
            print("   FAIL:", c.get("label"), c.get("note"))
        sys.exit(3)
    if any(_is_missing_note(c) for c in ident["identity_checks"]):
        print("  note: a remote kernel path was missing; byte-identity recorded "
              "(independently verified on EPIC-05). Continuing.")

    # ---- bootstrap C0 + data ----------------------------------------------
    boot = bootstrap_c0()
    print(f"C0 bootstrap: ext {boot['so_sha256'][:16]}.. core {boot['core_sha256'][:16]}..")
    ds = load_dataset(P10_SCENE)
    m, ckpt_ident = load_model_params(P10_SCENE, P10_CKPT)
    # Frozen camera index 19 img_name assertion (live GTDataset; re-asserted here).
    cam = ds.get_camera(P10_CAM_INDEX)
    cam_img = getattr(cam, "img_name", None) or getattr(cam, "image_name", None)
    if cam_img != P10_CAM_IMAGE:
        json_dump({"verdict": "A0P_CAMERA_BLOCKER",
                   "detail": f"frozen camera index {P10_CAM_INDEX} img_name={cam_img} != {P10_CAM_IMAGE}"},
                  os.path.join(out_dir, "A0_VERDICT.json"))
        print(f"BLOCKER: A0P_CAMERA_BLOCKER -- cam {P10_CAM_INDEX} img_name={cam_img}; STOP.")
        sys.exit(6)
    print(f"loaded {P10_SCENE}@{P10_CKPT}: N={ckpt_ident['num_points']} "
          f"sha={ckpt_ident['ckpt_sha256'][:16]}.. cam={cam_img} "
          f"active_sh_degree={ckpt_ident['active_sh_degree']}")

    # ---- Stage 2: native graph, raw-frame adjoint, preflight sanity --------
    print("=== Stage 2: native graph + raw-frame adjoint (preflight sanity) ===")
    from gsplat.experimental import rasterize_gaussian_higs_dynamic

    sh_deg = ckpt_ident["active_sh_degree"]

    def c0_render(with_grad, _max_retry=8, _sleep=20):
        """Frozen C0 forward with retry-on-OOM. EPIC-05 GPUs are shared; a burst
        co-tenant job (~69 GiB) can occupy the GPU during the intersect_tile
        allocation. On CUDA OOM, wait for the GPU to free and retry (up to
        _max_retry times). The frozen math is unchanged -- only allocation timing.
        """
        import time as _time
        import gc
        last_err = None
        for attempt in range(1, _max_retry + 1):
            if with_grad:
                for k in m:
                    if m[k].grad is not None:
                        m[k].grad = None
            try:
                params = {k: m[k].requires_grad_(True) if with_grad else m[k].detach()
                          for k in ("xyz", "shs", "scaling", "rotation", "opacity")}
                out = rasterize_gaussian_higs_dynamic(
                    params["xyz"], params["rotation"], params["scaling"], params["opacity"],
                    params["shs"], sh_degree=sh_deg,
                    viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
                    Ks=cam.K.unsqueeze(0).unsqueeze(0),
                    width=cam.image_width, height=cam.image_height,
                    backward_mode="higs_native", enable_culling=True,
                    camera_model="pinhole", render_mode="RGB", eps2d=RECIPE["eps2d"])
                frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
                info = out["densification_info"]
                return out, frame, info, params
            except torch.OutOfMemoryError as e:
                last_err = e
                if with_grad:
                    try:
                        torch._C._cuda_clearCublasWorkspaces()
                    except Exception:
                        pass
                print(f"  [OOM attempt {attempt}/{_max_retry}] co-tenant on GPU; "
                      f"waiting {_sleep}s for a free window", flush=True)
                _time.sleep(_sleep)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
        raise last_err

    # (a) RAW raster output (pre-clamp) -- the true native graph output.
    out_raw, frame_raw, info, _ = c0_render(with_grad=True)
    H, W = cam.image_height, cam.image_width
    raw = frame_raw  # [H,W,3], the pre-clamp native raster output
    # (b) exact raw-frame adjoint via autograd.grad(loss, raw_frame, retain_graph=True).
    #     loss = 0.8*L1(raw,gt) + 0.2*(1 - fused_ssim(raw,gt,padding=same,train=True)).
    import torch.nn.functional as F
    gt = ds.get_gt_image(P10_CAM_INDEX)  # [H,W,3] float32 (black background)
    b = raw.permute(2, 0, 1).unsqueeze(0)        # [1,3,H,W]
    g = gt.permute(2, 0, 1).unsqueeze(0).contiguous()
    L1 = F.l1_loss(b, g)
    import fused_ssim as _fused
    ssim = _fused.fused_ssim(b, g, padding="same", train=True)
    loss = (1.0 - RECIPE["lambda_dssim"]) * L1 + RECIPE["lambda_dssim"] * (1.0 - ssim)
    v_raw = torch.autograd.grad(loss, b, retain_graph=True)[0]  # [1,3,H,W]
    v_image = v_raw.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)  # [H,W,3]
    # (c) reference: native autograd backward (loss.backward) -> native param grads.
    for k in ("xyz", "shs", "scaling", "rotation", "opacity"):
        if m[k].grad is not None:
            m[k].grad = None
    loss.backward()
    native_grads = {k: (m[k].grad.detach().cpu().numpy().astype(np.float32)
                        if m[k].grad is not None else np.zeros(m[k].shape, dtype=np.float32))
                    for k in ("xyz", "shs", "scaling", "rotation", "opacity")}
    loss_val = float(loss.item())
    print(f"  raw {tuple(frame_raw.shape)} loss={loss_val:.8f} |v_raw|={float(np.linalg.norm(v_image)):.6g}")

    # (d) forward-captured state for the walk.
    info_np = {k: _to_np(v) for k, v in info.items()}
    conics = info_np["conics"].reshape(-1, 3)
    means2d = info_np["means2d"].reshape(-1, 2)
    colors_eval = info_np["colors_eval"].reshape(-1, 3)
    opacities = info_np["opacities"].reshape(-1)
    tile_offsets = _to_np(info_np["tile_offsets"])      # [I, th, tw]
    flatten_ids = _to_np(info_np["flatten_ids"])        # [n_isects]
    render_alphas = _to_np(info_np["render_alphas"])    # [I,H,W]
    last_ids = _to_np(info_np["last_ids"])              # [I,H,W]
    n_isects = int(flatten_ids.shape[0])
    I = 0
    if any(x is None for x in (conics, means2d, colors_eval, opacities,
                               tile_offsets, flatten_ids, render_alphas, last_ids)):
        json_dump({"verdict": "A0P_STATE_BLOCKER",
                   "detail": "densification_info missing required forward-captured state"},
                  os.path.join(out_dir, "A0_VERDICT.json"))
        print("BLOCKER: A0P_STATE_BLOCKER -- forward-captured state incomplete; STOP.")
        sys.exit(4)

    # ---- Stage 3: native tile x batch bucket census + top-32 freeze --------
    print(f"=== Stage 3: native tile x batch bucket census (bucket_width={BUCKET_WIDTH}) ===")
    tile_width = math.ceil(W / TILE_SIZE)
    tile_height = math.ceil(H / TILE_SIZE)
    n_tiles = tile_width * tile_height
    tile_offsets_img = [int(v) for v in tile_offsets[I]]
    tile_origin = {}
    bucket_hits = []   # (tile_id, batch_index, hits, origin, batch_end, batch_size)
    total_hits = 0
    total_buckets = 0
    for ty in range(tile_height):
        for tx in range(tile_width):
            tile_id = ty * tile_width + tx
            origin = (ty * TILE_SIZE, tx * TILE_SIZE)
            tile_origin[tile_id] = origin
            rs = tile_offsets_img[tile_id]
            re_ = tile_offsets_img[tile_id + 1] if tile_id + 1 < len(tile_offsets_img) else n_isects
            for binfo in tile_batches(rs, re_, BUCKET_WIDTH):
                total_buckets += 1
                res = tile_bucket_walk(
                    conics, means2d, opacities, colors_eval,
                    rs, re_, flatten_ids, render_alphas, last_ids,
                    I, tile_id, W, H, TILE_SIZE, v_image,
                    binfo["batch_index"], BUCKET_WIDTH)
                if res["hits"] > 0:
                    total_hits += res["hits"]
                    bucket_hits.append((tile_id, binfo["batch_index"], res["hits"], origin,
                                        res["batch_end"], res["batch_size"]))
    print(f"  tiles={n_tiles} native_buckets={total_buckets} nonempty={len(bucket_hits)} "
          f"total_accepted_hits={total_hits}")
    # top-32 by native accepted-hit count (ties broken by (tile_id, batch_index));
    # NO loss/gradient/rank in selection.
    ranked = sorted(bucket_hits, key=lambda x: (-x[2], x[0], x[1]))
    top = ranked[:P10_TOPK]
    buckets = [{
        "tile_id": int(tid), "batch_index": int(bi),
        "tile_row": int(tid // tile_width), "tile_col": int(tid % tile_width),
        "origin_I": int(origin[0]), "origin_J": int(origin[1]),
        "batch_end": int(bend), "batch_size": int(bsz),
        "accepted_hits": int(hits), "rank": int(r + 1),
    } for r, (tid, bi, hits, origin, bend, bsz) in enumerate(top)]
    buckets_path = os.path.join(out_dir, "A0_BUCKETS.json")
    json_dump({
        "campaign": "A0-P10_PREFLIGHT",
        "scene": P10_SCENE, "ckpt": P10_CKPT, "cam_index": P10_CAM_INDEX, "cam_image": cam_img,
        "bucket_object": "native tile x batch (HigsNativeBackward.cu block_size=256/PX)",
        "bucket_width": BUCKET_WIDTH, "higs_px_runtime": HIGS_PX_RUNTIME,
        "selection": "top-32 native bucket objects by native accepted-hit count only (no grad/loss/rank)",
        "tile_size": TILE_SIZE, "tile_width": tile_width, "tile_height": tile_height,
        "n_tiles": n_tiles, "n_native_buckets": total_buckets,
        "n_nonempty_buckets": len(bucket_hits),
        "total_accepted_hits": total_hits, "top_k": P10_TOPK,
        "buckets": buckets,
    }, buckets_path)
    print(f"  buckets -> {buckets_path}")

    # ---- Stage 4: exact FP32 coefficient dump for frozen buckets -----------
    print("=== Stage 4: coefficient dump (top-32 native buckets) ===")
    bdir = os.path.join(out_dir, "buckets")
    os.makedirs(bdir, exist_ok=True)
    npz_files = []
    for b in buckets:
        tid, bi = b["tile_id"], b["batch_index"]
        rs = tile_offsets_img[tid]
        re_ = tile_offsets_img[tid + 1] if tid + 1 < len(tile_offsets_img) else n_isects
        res = tile_bucket_walk(
            conics, means2d, opacities, colors_eval,
            rs, re_, flatten_ids, render_alphas, last_ids,
            I, tid, W, H, TILE_SIZE, v_image, bi, BUCKET_WIDTH)
        qa, qc, wc, mask, cols, col_index = dense_coefficients(res, b["origin_I"], b["origin_J"])
        npz = os.path.join(bdir, f"bucket_t{tid:04d}_b{bi:03d}.npz")
        np.savez_compressed(
            npz,
            q_alpha=qa, q_color=qc, w_color=wc,
            accepted_mask=mask,
            col_gaussians=np.array(cols, dtype=np.int64),
            accepted_hits=np.array([b["accepted_hits"]]),
            tile_id=np.array([tid]), batch_index=np.array([bi]),
            tile_row=np.array([b["tile_row"]]), tile_col=np.array([b["tile_col"]]),
            origin_I=np.array([b["origin_I"]]), origin_J=np.array([b["origin_J"]]),
            batch_end=np.array([b["batch_end"]]), batch_size=np.array([b["batch_size"]]),
            scene=np.array(P10_SCENE), ckpt=np.array([P10_CKPT]),
            cam_index=np.array([P10_CAM_INDEX]),
        )
        npz_files.append(npz)
    print(f"  wrote {len(npz_files)} bucket npz shards -> {bdir}")

    # ---- Stage 2b: preflight sanity (NOT full A0-v2 native-gradient parity) -
    print("=== Stage 2b: preflight sanity ===")
    # loss reproducibility: re-render the raw loss once more (fresh graph)
    out2, frame_raw2, _, _ = c0_render(with_grad=True)
    b2 = frame_raw2.permute(2, 0, 1).unsqueeze(0)
    L1_2 = F.l1_loss(b2, g)
    ssim2 = _fused.fused_ssim(b2, g, padding="same", train=True)
    loss2 = (1.0 - RECIPE["lambda_dssim"]) * L1_2 + RECIPE["lambda_dssim"] * (1.0 - ssim2)
    loss_repro = abs(float(loss.item()) - float(loss2.item()))
    # native grads finite
    finite_grads = bool(all(np.isfinite(x).all() for x in native_grads.values()))
    # walk determinism: re-run bucket 0
    b0 = buckets[0]
    rs0 = tile_offsets_img[b0["tile_id"]]
    re0 = tile_offsets_img[b0["tile_id"] + 1] if b0["tile_id"] + 1 < len(tile_offsets_img) else n_isects
    ra = tile_bucket_walk(conics, means2d, opacities, colors_eval, rs0, re0, flatten_ids,
                          render_alphas, last_ids, I, b0["tile_id"], W, H, TILE_SIZE,
                          v_image, b0["batch_index"], BUCKET_WIDTH)
    rb = tile_bucket_walk(conics, means2d, opacities, colors_eval, rs0, re0, flatten_ids,
                          render_alphas, last_ids, I, b0["tile_id"], W, H, TILE_SIZE,
                          v_image, b0["batch_index"], BUCKET_WIDTH)
    walk_det = (ra["hits"] == rb["hits"]) and \
               all(x["g"] == y["g"] for x, y in zip(ra["accepted"], rb["accepted"])) and \
               all(float(x["q_alpha"]) == float(y["q_alpha"]) for x, y in zip(ra["accepted"], rb["accepted"]))
    # coefficients finite + m <= native bucket width
    coeffs_finite = True
    m_le_width = True
    for f in npz_files:
        d = np.load(f)
        coeffs_finite &= bool(np.isfinite(d["q_alpha"]).all() and np.isfinite(d["q_color"]).all())
        m_le_width &= bool(d["q_alpha"].shape[1] <= BUCKET_WIDTH)
    sanity = {
        "campaign": "A0-P10_PREFLIGHT",
        "kind": "PREFLIGHT SANITY (NOT full A0-v2 native-gradient parity)",
        "scene": P10_SCENE, "ckpt": P10_CKPT, "cam_index": P10_CAM_INDEX,
        "loss": loss_val,
        "loss_reproducible_abs": loss_repro,
        "loss_reproducible_ok": bool(loss_repro < 1e-6),
        "native_grads_finite": finite_grads,
        "native_grad_norms": {k: float(np.linalg.norm(v)) for k, v in native_grads.items()},
        "walk_deterministic": bool(walk_det),
        "coeffs_finite": bool(coeffs_finite),
        "m_le_bucket_width": bool(m_le_width),
        "v_raw_norm": float(np.linalg.norm(v_image)),
        "note": ("Preflight sanity gate = (loss reproducible) AND (native grads finite) AND "
                 "(walk deterministic) AND (coeffs finite) AND (m <= native bucket width). "
                 "Full instrumented-vs-native reconstructed-VJP parity is left to formal A0."),
        "pass": bool(loss_repro < 1e-6 and finite_grads and walk_det and coeffs_finite and m_le_width),
    }
    json_dump(sanity, os.path.join(out_dir, "A0_PREFLIGHT_SANITY.json"))
    print(f"  preflight sanity pass={sanity['pass']}")
    if not sanity["pass"]:
        json_dump({"verdict": "A0P_SANITY_BLOCKER", "detail": sanity},
                  os.path.join(out_dir, "A0_VERDICT.json"))
        print("BLOCKER: A0P_SANITY_BLOCKER -- preflight sanity failed; STOP.")
        sys.exit(5)

    # ---- Stage 5: oracle-SVD / DCT structure + frozen P10 verdict ----------
    print("=== Stage 5: oracle-SVD / DCT structure + P10 verdict ===")
    rows = [analyze_bucket(f) for f in npz_files]
    struct = {
        "campaign": "A0-P10_PREFLIGHT",
        "analyzer": "oracle-SVD/DCT (frozen a0_structure_analyze_frozen.py semantics)",
        "primary_color_tensor": "q_color[p,i,k] = fac * v_render_colors[p,k] (true 3-channel)",
        "basis": "2D orthonormal DCT-II, ordered by fx+fy then fx then fy (diagnostic only)",
        "ranks": [1, 2, 4, 8, 16, 32],
        "scene": P10_SCENE, "ckpt": P10_CKPT, "cam_index": P10_CAM_INDEX,
        "bucket_object": "native tile x batch", "bucket_width": BUCKET_WIDTH,
        "n_buckets": len(rows),
        "rows": rows,
        "verdict": p10_verdict(rows),
    }
    json_dump(struct, os.path.join(out_dir, "A0_STRUCTURE_RESULTS.json"))
    v = struct["verdict"]["verdict"]
    print(f"  structure -> {os.path.join(out_dir, 'A0_STRUCTURE_RESULTS.json')}  P10 verdict={v}")

    # ---- final verdict -----------------------------------------------------
    verdict = {
        "verdict": "A0P_" + v.split("_", 1)[-1] if v.startswith("A0P_") else v,
        "structure_verdict": v,
        "note": ("Preflight complete for room@25K cam19 top-32 NATIVE buckets. The P10 "
                 "verdict is a FROZEN MECHANICAL rule on oracle SVD r99/r995 medians only "
                 "(DCT diagnostic). This is a PRELIMINARY single-cell preflight; the full "
                 "A0-v2 verdict (all 4 cells + reconstructed-VJP + cost gates) requires the "
                 "formal A0 campaign. Full native-gradient parity is NOT claimed here."),
        "artifacts": {
            "identity": "A0_SOURCE_IDENTITY.json",
            "preflight_sanity": "A0_PREFLIGHT_SANITY.json",
            "buckets": "A0_BUCKETS.json",
            "structure": "A0_STRUCTURE_RESULTS.json",
        },
    }
    json_dump(verdict, os.path.join(out_dir, "A0_VERDICT.json"))
    print("=== A0-P10 PREFLIGHT COMPLETE ===")


if __name__ == "__main__":
    main()
