#!/usr/bin/env python
"""g5a_extract_gt.py v2 - recover per-tile (ti,tj) + REAL GT image patches for G5-A.

The frozen workload.npz DROPPED ti/tj, and re-running extract_tiles does NOT
reproduce the frozen tile order (gsplat tile binning is not bit-reproducible across
runs). So we recover (ti,tj) FROM THE WORKLOAD ALONE, self-validating against the
frozen teacher render ref_teacher_f64[t][:,:3] (the 3DGS alpha-composite at the
tile exact 256 pixels):

  For each view, build the FULL tile-grid GT patch matrix P (N=th*tw, 256, 3) and
  per-tile squared norms. For each workload tile t in the view, the teacher-GT MSE
  over every grid cell is
      mse[k] = mean(P[k]^2) - 2<P[k],T>/768 + mean(T^2),   T = teacher[t].rgb.vec
  (exact expansion of mean((P[k]-T)^2)). Greedy INJECTIVE assignment (most-
  confident-first) maps each teacher to its best GT cell, enforcing distinct (ti,tj).
  assign is stored INDEXED BY ORIGINAL TILE ROW (v1 bug: indexed by sorted order ->
  saved patches were permuted vs teacher renders; fixed here).

Correctness is OBJECTIVE and self-checking: teacher render == GT at the true cell,
so the recovered cell yields high teacher-GT PSNR. GT patch = img[ti*16:(ti+1)*16,
tj*16:(tj+1)*16,:3]/255 (row-major 256x3), aligned to the workload px order. Pure
numpy + PIL. Consumes the FROZEN workload read-only; produces the real-GT target for
G5-A (goal sec.5: optimize from real GT, no teacher-derived target in native loss).
"""
from __future__ import annotations
import json, os, time
import numpy as np
import sys
sys.path.insert(0, "/home/liaoyuanjun/cgt_realscene/final/SCRIPTS")
from g1_compressibility import read_images_bin, TILE  # noqa: E402

DATA = "/tmp/3dgs-renderer-benchmark/data"
WORKLOAD = "/home/liaoyuanjun/cgt_f0_fused_cuda/results/workload.npz"
OUTDIR = "/home/liaoyuanjun/cgt_g5_native/results"
SCENES = ["room", "garden", "bicycle"]


def load_gt_padded(scene, name):
    from PIL import Image
    p = f"{DATA}/datasets/mipnerf360/{scene}/images/{name}"
    img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    H, W = img.shape[0], img.shape[1]
    pad_h = (-H) % TILE; pad_w = (-W) % TILE
    edge = (pad_h > 0 or pad_w > 0)
    if edge:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return np.ascontiguousarray(img, dtype=np.float32), int(H), int(W), bool(edge)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = np.load(WORKLOAD)
    sc_arr = d["scene"]; view_arr = d["view"]; off = d["tile_off"]
    teacher = d["ref_teacher_f64"]; n_tiles = int(d["n_tiles"])
    NPX = TILE * TILE
    val = {"per_scene": {}, "n_tiles": n_tiles, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "teacher_gt_psnr": [], "teacher_gt_corr": [], "below_15dB": 0, "below_25dB": 0}
    gt_all, ti_all, tj_all, view_all, scene_all, edge_all = [], [], [], [], [], []
    t0 = time.time()
    for scene in SCENES:
        imgs = read_images_bin(f"{DATA}/datasets/mipnerf360/{scene}/sparse/0/images.bin")
        idx = np.where(sc_arr == scene)[0]
        views = []
        for v in view_arr[idx]:
            if int(v) not in views: views.append(int(v))
        rec = {"n_tiles": int(len(idx)), "views": views}
        psnr_scene, corr_scene = [], []
        for vi in views:
            sidx = idx[view_arr[idx] == vi]
            iid, name, cam, qv, tv = imgs[vi]
            gt, H, W, edge = load_gt_padded(scene, name)
            th, tw = gt.shape[0] // TILE, gt.shape[1] // TILE
            P = gt.reshape(th, TILE, tw, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(th * tw, NPX, 3).astype(np.float64)
            P2 = P.reshape(th * tw, NPX * 3)
            Psq = (P2 ** 2).mean(axis=1)
            # teacher-GT MSE matrix (n_view_tiles, N)
            M = np.empty((len(sidx), th * tw), dtype=np.float64)
            Tv = np.empty((len(sidx), NPX * 3), dtype=np.float64)
            for r, wt in enumerate(sidx):
                T = teacher[wt][:, :3].reshape(-1)
                Tv[r] = T
                M[r] = Psq - 2.0 * (P2 @ T) / NPX / 3.0 + float((T ** 2).mean())
            # greedy injective assignment, most-confident-first; INDEX BY ORIGINAL ROW
            order = np.argsort(M.min(axis=1))
            taken = set(); assign = [None] * len(sidx)
            for r in order:
                col = M[r].copy()
                for k in taken: col[k] = np.inf
                b = int(np.argmin(col)); taken.add(b); assign[int(r)] = (b // tw, b % tw, b, float(M[r, b]))
            for r, wt in enumerate(sidx):
                ti, tj, b, mse = assign[r]
                T = Tv[r]
                pflat = P2[b]
                pc = float(np.corrcoef(pflat, T)[0, 1]) if pflat.std() > 1e-8 and T.std() > 1e-8 else 1.0
                psnr = float(-10.0 * np.log10(max(mse, 1e-30)))
                psnr_scene.append(psnr); corr_scene.append(pc)
                val["teacher_gt_psnr"].append(psnr); val["teacher_gt_corr"].append(pc)
                patch = P[b].reshape(TILE, TILE, 3).astype(np.float32)
                gt_all.append(np.ascontiguousarray(patch))
                ti_all.append(ti); tj_all.append(tj); view_all.append(int(vi))
                scene_all.append(scene); edge_all.append(bool(edge))
        rec["teacher_gt_psnr_mean"] = float(np.mean(psnr_scene))
        rec["teacher_gt_psnr_min"] = float(np.min(psnr_scene))
        rec["teacher_gt_psnr_max"] = float(np.max(psnr_scene))
        rec["teacher_gt_corr_mean"] = float(np.mean(corr_scene))
        rec["teacher_gt_corr_min"] = float(np.min(corr_scene))
        rec["n_below_15dB"] = int(sum(1 for p in psnr_scene if p < 15.0))
        rec["n_below_25dB"] = int(sum(1 for p in psnr_scene if p < 25.0))
        val["per_scene"][scene] = rec
        print(f"[{scene}] tiles={len(sidx)*len(views)} psnr mean={rec['teacher_gt_psnr_mean']:.2f} "
              f"min={rec['teacher_gt_psnr_min']:.2f} corr mean={rec['teacher_gt_corr_mean']:.4f} "
              f"min={rec['teacher_gt_corr_min']:.4f} below15dB={rec['n_below_15dB']} below25dB={rec['n_below_25dB']}", flush=True)

    val["teacher_gt_psnr_mean"] = float(np.mean(val["teacher_gt_psnr"]))
    val["teacher_gt_psnr_median"] = float(np.median(val["teacher_gt_psnr"]))
    val["teacher_gt_psnr_min"] = float(np.min(val["teacher_gt_psnr"]))
    val["teacher_gt_corr_mean"] = float(np.mean(val["teacher_gt_corr"]))
    val["teacher_gt_corr_min"] = float(np.min(val["teacher_gt_corr"]))
    val["below_15dB"] = int(sum(1 for p in val["teacher_gt_psnr"] if p < 15.0))
    val["below_25dB"] = int(sum(1 for p in val["teacher_gt_psnr"] if p < 25.0))
    val["below_30dB"] = int(sum(1 for p in val["teacher_gt_psnr"] if p < 30.0))
    val["elapsed_s"] = round(time.time() - t0, 2)

    gt_arr = np.ascontiguousarray(np.stack(gt_all, axis=0).reshape(len(gt_all), NPX, 3).astype(np.float32))
    out = os.path.join(OUTDIR, "g5a_gt_patches.npz")
    np.savez(out, gt=gt_arr,
             ti=np.asarray(ti_all, dtype=np.int32), tj=np.asarray(tj_all, dtype=np.int32),
             view=np.asarray(view_all, dtype=np.int32), scene=np.array(scene_all),
             edge=np.asarray(edge_all, dtype=bool))
    with open(os.path.join(OUTDIR, "g5a_gt_extract_validation.json"), "w") as f:
        json.dump(val, f, indent=2)
    print("WROTE", out, "gt", gt_arr.shape, flush=True)
    print(json.dumps({"teacher_gt_psnr_mean": val["teacher_gt_psnr_mean"],
                      "teacher_gt_psnr_median": val["teacher_gt_psnr_median"],
                      "teacher_gt_psnr_min": val["teacher_gt_psnr_min"],
                      "teacher_gt_corr_mean": val["teacher_gt_corr_mean"],
                      "teacher_gt_corr_min": val["teacher_gt_corr_min"],
                      "below_15dB": val["below_15dB"], "below_25dB": val["below_25dB"],
                      "below_30dB": val["below_30dB"],
                      "elapsed_s": val["elapsed_s"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
