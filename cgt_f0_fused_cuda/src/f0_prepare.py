#!/usr/bin/env python3
"""CGT-F0 prepare: build the kernel-ready mixed workload + trusted references.

For each scene (room/garden/bicycle), deterministically:
  1. extract a representative, work-decile-stratified set of gsplat tiles,
  2. compute the corrected teacher (M-Gaussian alpha composite) over 16x16,
  3. fit the rank-R CGT student (PRIMARY rank frozen at R=4 from upstream evidence),
  4. run the EXACT frozen rigorous interval certificate (certify_tile) -> per-tile
     safe label (passed = rigorous worst-case max-abs <= GATE) + fitted coeffs,
  5. emit kernel-ready flattened arrays AND float64/float32 trusted references.

Reuses the frozen upstream math verbatim (no re-implementation):
  g1_compressibility.tile_teacher / fit_cgt / predict_and_vjp
  g2_rigorous_certificate.certify_tile / extract_tiles
Correctness is decided later by the harness against these references.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

SCRIPTS = "/home/liaoyuanjun/cgt_realscene/final/SCRIPTS"
sys.path.insert(0, SCRIPTS)
import torch  # noqa: E402
import gsplat  # noqa: E402  (needed by extract_tiles)
from g1_compressibility import (read_cameras_bin, read_images_bin, q2R,  # noqa: E402
                                tile_teacher, fit_cgt, predict_and_vjp,
                                TILE, POWER_CLIP, GATE_MAX_ABS)
from g2_rigorous_certificate import certify_tile, extract_tiles  # noqa: E402

DATA = "/tmp/3dgs-renderer-benchmark/data"
# Checkpoints match the frozen G2 rigorous-certificate evidence (representative, validated upstream).
SCENES = [
    ("room",    "/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_multiphase_v2/checkpoints/iter_15000.pt"),
    ("garden",  "/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_garden_30k/checkpoints/iter_30000.pt"),
    ("bicycle", "/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_bicycle_30k/checkpoints/iter_30000.pt"),
]
NPX = TILE * TILE  # 256


def shared_px():
    """Centered 16x16 pixel coords, IDENTICAL for every tile."""
    xs = np.arange(TILE) + 0.5
    ys = np.arange(TILE) + 0.5
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    px = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    cx0, cy0 = px.mean(0)
    px[:, 0] -= cx0
    px[:, 1] -= cy0
    return np.ascontiguousarray(px, dtype=np.float64)


def refs_for_tile(px, gids, m2d, con, op, cl, coeffs):
    """float64 (authoritative) and float32 (fast-path) trusted references."""
    out = {}
    for dt in (np.float64, np.float32):
        pxd = px.astype(dt); m2 = m2d.astype(dt); co = con.astype(dt)
        o = op.astype(dt); c = cl.astype(dt); gid = gids.astype(np.int64)
        E, T = tile_teacher(pxd[:, 0:1], pxd[:, 1:2], gid, m2, co, o, c)
        ref_teacher = np.concatenate([E, T], axis=1).astype(dt)
        ref_student = predict_and_vjp(coeffs.astype(dt), pxd).astype(dt)
        out["teacher_" + dt.__name__] = np.ascontiguousarray(ref_teacher)
        out["student_" + dt.__name__] = np.ascontiguousarray(ref_student)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/liaoyuanjun/cgt_f0_fused_cuda/results/workload.npz")
    ap.add_argument("--views", type=int, default=5, help="views sampled per scene")
    ap.add_argument("--tiles-per-view", type=int, default=10, help="tiles sampled per view (decile-stratified)")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--min-size", type=float, default=1.0)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--rank", type=int, default=4, help="PRIMARY rank (frozen from upstream evidence)")
    ap.add_argument("--diag-tiles", type=int, default=10, help="tiles/scene for R8/R16 diagnostic fit")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()
    validate = not args.no_validate
    dev = "cuda"

    px_shared = shared_px()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    all_means2d, all_conics, all_opac, all_colors = [], [], [], []
    tile_off = [0]
    coeffs_list, safe_list = [], []
    scene_l, view_l, ti_l, tj_l, M_l, rig_l, leaves_l = [], [], [], [], [], [], []
    ref_teach_f64, ref_stud_f64, ref_teach_f32, ref_stud_f32 = [], [], [], []
    diag = []

    rng_views = np.random.default_rng(args.seed)
    t_start = time.time()

    for scene, ckpt in SCENES:
        cams = read_cameras_bin(f"{DATA}/datasets/mipnerf360/{scene}/sparse/0/cameras.bin")
        imgs = read_images_bin(f"{DATA}/datasets/mipnerf360/{scene}/sparse/0/images.bin")
        n_views = len(imgs)
        view_idx = sorted(rng_views.choice(n_views, size=min(args.views, n_views), replace=False).tolist())
        print(f"[{scene}] ckpt={os.path.basename(ckpt)} views={view_idx}", flush=True)

        scene_tiles = 0
        for vi in view_idx:
            seed = args.seed + vi * 101
            try:
                tiles = extract_tiles(scene, ckpt, DATA, int(vi), args.tiles_per_view, seed)
            except Exception as ex:
                import traceback; traceback.print_exc()
                print(f"[{scene}] view {vi} extract FAILED: {ex}", flush=True)
                continue
            if not tiles:
                print(f"[{scene}] view {vi} no tiles", flush=True)
                continue
            m2d = tiles[0]["means2d"]; con = tiles[0]["conics"]
            op = tiles[0]["opac"]; cl = tiles[0]["colors"]
            for tt in tiles:
                gids = tt["gids"].astype(np.int64)
                px = tt["px"].astype(np.float64)
                M = int(tt["M"])
                E, T = tile_teacher(px[:, 0:1], px[:, 1:2], gids, m2d, con, op, cl)
                y = np.concatenate([E, T], axis=1)
                P = y.shape[0]
                tr = np.arange(0, P, 2)
                qt = np.column_stack([px[tr, 0], px[tr, 1]]).astype(np.float64)
                yt = y[tr].astype(np.float64)
                fit_seed = args.seed + scene_tiles * 1000 + args.rank
                coeffs = fit_cgt(qt, yt, args.rank, fit_seed, args.iterations).astype(np.float64)
                ctx = (gids, m2d, con, op, cl, coeffs)
                cert = certify_tile(tt, coeffs, args.min_size, args.max_depth,
                                    np.random.default_rng(fit_seed + 7), validate)
                safe = int(bool(cert["passed"]))

                rr = refs_for_tile(px, gids, m2d, con, op, cl, coeffs)

                gm = m2d[gids]; gc = con[gids]; go = op[gids]; gcl = cl[gids]
                all_means2d.append(np.ascontiguousarray(gm, dtype=np.float64))
                all_conics.append(np.ascontiguousarray(gc, dtype=np.float64))
                all_opac.append(np.ascontiguousarray(go, dtype=np.float64))
                all_colors.append(np.ascontiguousarray(gcl, dtype=np.float64))
                tile_off.append(tile_off[-1] + M)

                coeffs_list.append(np.ascontiguousarray(coeffs, dtype=np.float64))
                safe_list.append(safe)
                scene_l.append(scene)
                view_l.append(int(vi))
                ti_l.append(int(tt.get("ti", -1)))
                tj_l.append(int(tt.get("tj", -1)))
                M_l.append(M)
                rig_l.append(float(cert["rigorous_max_abs"]))
                leaves_l.append(int(cert["n_leaves"]))
                ref_teach_f64.append(rr["teacher_float64"])
                ref_stud_f64.append(rr["student_float64"])
                ref_teach_f32.append(rr["teacher_float32"])
                ref_stud_f32.append(rr["student_float32"])
                scene_tiles += 1

                if scene_tiles <= args.diag_tiles:
                    for rk in (8, 16):
                        cs = int(rk)
                        cd = fit_cgt(qt, yt, cs, fit_seed + cs, args.iterations).astype(np.float64)
                        ct = certify_tile(tt, cd, args.min_size, args.max_depth,
                                          np.random.default_rng(fit_seed + cs + 7), validate)
                        diag.append(dict(scene=scene, view=int(vi), M=M, rank=cs,
                                         passed=int(bool(ct["passed"])),
                                         rigorous_max_abs=float(ct["rigorous_max_abs"]),
                                         n_leaves=int(ct["n_leaves"])))
                if scene_tiles % 10 == 0:
                    print(f"[{scene}] tiles so far={scene_tiles} t={time.time()-t_start:.1f}s", flush=True)
        print(f"[{scene}] DONE tiles={scene_tiles} t={time.time()-t_start:.1f}s", flush=True)

    Mt = tile_off[-1]
    n_tiles = len(safe_list)
    R = args.rank
    means2d_all = np.ascontiguousarray(np.concatenate(all_means2d, axis=0), dtype=np.float64) if Mt else np.zeros((0, 2))
    conics_all = np.ascontiguousarray(np.concatenate(all_conics, axis=0), dtype=np.float64) if Mt else np.zeros((0, 3))
    opac_all = np.ascontiguousarray(np.concatenate(all_opac, axis=0), dtype=np.float64) if Mt else np.zeros((0,))
    colors_all = np.ascontiguousarray(np.concatenate(all_colors, axis=0), dtype=np.float64) if Mt else np.zeros((0, 3))
    coeffs_all = np.ascontiguousarray(np.stack(coeffs_list, axis=0), dtype=np.float64).reshape(n_tiles, R, 10)
    safe_arr = np.asarray(safe_list, dtype=np.int32)
    off_arr = np.asarray(tile_off, dtype=np.int32)
    px_arr = np.ascontiguousarray(px_shared, dtype=np.float64)
    ref_teach_f64_a = np.ascontiguousarray(np.stack(ref_teach_f64, axis=0)) if n_tiles else np.zeros((0, NPX, 4))
    ref_stud_f64_a = np.ascontiguousarray(np.stack(ref_stud_f64, axis=0)) if n_tiles else np.zeros((0, NPX, 4))
    ref_teach_f32_a = np.ascontiguousarray(np.stack(ref_teach_f32, axis=0)).astype(np.float32) if n_tiles else np.zeros((0, NPX, 4), np.float32)
    ref_stud_f32_a = np.ascontiguousarray(np.stack(ref_stud_f32, axis=0)).astype(np.float32) if n_tiles else np.zeros((0, NPX, 4), np.float32)

    np.savez(args.out,
             means2d=means2d_all, conics=conics_all, opac=opac_all, colors=colors_all,
             tile_off=off_arr, px=px_arr, coeffs=coeffs_all, safe=safe_arr,
             R=np.int32(R), n_tiles=np.int32(n_tiles), Mt=np.int32(Mt),
             scene=np.array(scene_l), view=np.array(view_l, dtype=np.int32),
             ti=np.array(ti_l, dtype=np.int32), tj=np.array(tj_l, dtype=np.int32),
             M=np.array(M_l, dtype=np.int32),
             rigorous_max_abs=np.asarray(rig_l, dtype=np.float64),
             n_leaves=np.asarray(leaves_l, dtype=np.int32),
             ref_teacher_f64=ref_teach_f64_a, ref_student_f64=ref_stud_f64_a,
             ref_teacher_f32=ref_teach_f32_a, ref_student_f32=ref_stud_f32_a)

    M_arr = np.asarray(M_l, dtype=np.float64)
    scene_arr = np.array(scene_l)
    summary = dict(
        n_tiles=int(n_tiles), Mt=int(Mt), R=int(R), gate_max_abs=float(GATE_MAX_ABS),
        per_scene={s: dict(n_tiles=int(np.sum(scene_arr == s)),
                           safe=int(np.sum(safe_arr[scene_arr == s])),
                           M_median=float(np.median(M_arr[scene_arr == s])) if np.any(scene_arr == s) else None)
                   for s, _ in SCENES},
        safe_fraction=float(np.mean(safe_arr)) if n_tiles else None,
        M_median=float(np.median(M_arr)) if n_tiles else None,
        M_mean=float(np.mean(M_arr)) if n_tiles else None,
        weighted_fallback_fraction=float(1.0 - np.sum(M_arr[safe_arr == 1]) / np.sum(M_arr)) if n_tiles and np.sum(M_arr) > 0 else None,
        diag_rank_records=len(diag),
        elapsed_s=round(time.time() - t_start, 2),
        seed=args.seed, views=args.views, tiles_per_view=args.tiles_per_view,
        iterations=args.iterations, min_size=args.min_size, max_depth=args.max_depth,
        validate=validate, checkpoints={s: c for s, c in SCENES},
    )
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.splitext(args.out)[0] + "_diag.json", "w") as f:
        json.dump(diag, f, indent=2)
    print("WROTE", args.out, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
