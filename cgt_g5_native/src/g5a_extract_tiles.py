#!/usr/bin/env python3
"""g5a_extract_tiles.py - VERBATIM copy of frozen g2_rigorous_certificate.extract_tiles
with ONE change: the per-tile output dict now also carries (ti, tj).

Provenance: copied byte-for-byte from
  /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g2_rigorous_certificate.py :: extract_tiles
The tile-selection logic (rng = default_rng(seed), decile stratification, sel) is
UNCHANGED, so the selected tiles and their ORDER are bit-identical to the frozen
f0_prepare workload. Only the returned dict gains ti=int(ti), tj=int(tj).
Upstream file is NOT modified (read-only preserved); this is an isolated copy.
"""
from __future__ import annotations
import numpy as np
import sys
sys.path.insert(0, "/home/liaoyuanjun/cgt_realscene/final/SCRIPTS")
from g1_compressibility import read_cameras_bin, read_images_bin, q2R, TILE  # noqa: E402


def extract_tiles_gt(scene, ckpt, data, view_idx, max_tiles, seed):
    import torch, gsplat
    cams = read_cameras_bin(f"{data}/datasets/mipnerf360/{scene}/sparse/0/cameras.bin")
    imgs = read_images_bin(f"{data}/datasets/mipnerf360/{scene}/sparse/0/images.bin")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    means = ck["xyz"].cuda().float(); scales = torch.exp(ck["scaling"].cuda().float())
    quats = ck["rotation"].cuda().float(); opacities = torch.sigmoid(ck["opacity"].cuda().float()); shs = ck["shs"].cuda().float()
    deg = int(ck["active_sh_degree"]); rng = np.random.default_rng(seed)
    iid, name, cam, qv, tv = imgs[view_idx]; mid, W, H, par = cams[cam]; fx, fy, cx, cy = par
    R_w2c = q2R(qv); t_w2c = np.array(tv); t_c2w = -R_w2c.T @ t_w2c
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")[None]
    viewmat[0, :3, :3] = torch.as_tensor(R_w2c, dtype=torch.float32, device="cuda")
    viewmat[0, :3, 3] = torch.as_tensor(t_w2c, dtype=torch.float32, device="cuda")
    K = torch.as_tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32, device="cuda")
    campos = torch.as_tensor(t_c2w, dtype=torch.float32, device="cuda")[None]
    rc, ra, meta = gsplat.rasterization(means, quats, scales, opacities, shs, viewmat, K, width=W, height=H,
                                        sh_degree=deg, render_mode="RGB", packed=True, tile_size=TILE)
    dirs = torch.nn.functional.normalize(means - campos, dim=-1)
    colors = (gsplat.spherical_harmonics(deg, dirs, shs) + 0.5).clamp(0, 1).detach().cpu().numpy()[meta["gaussian_ids"].detach().cpu().numpy()]
    def flat2(x):
        x = meta[x].detach().cpu().numpy(); return x[0] if x.ndim == 3 else x
    means2d = flat2("means2d"); conics = flat2("conics"); opac = flat2("opacities")
    io = meta["isect_offsets"].detach().cpu().numpy().reshape(-1); fi = meta["flatten_ids"].detach().cpu().numpy()
    th, tw = int(meta["tile_height"]), int(meta["tile_width"]); total = len(fi)
    starts = io.astype(np.int64)
    def end_of(ti, tj):
        if tj + 1 < tw: return int(starts[ti*tw + tj + 1])
        if ti + 1 < th: return int(starts[(ti + 1)*tw + 0])
        return total
    tiles = []
    for ti in range(th):
        for tj in range(tw):
            s = int(starts[ti*tw + tj]); e = end_of(ti, tj)
            if e - s > 0: tiles.append((ti, tj, s, e))
    counts = np.array([e - s for (_, _, s, e) in tiles])
    order = np.argsort(counts); ranks_sel = np.empty(len(tiles), dtype=np.int64); ranks_sel[order] = np.arange(len(tiles))
    decile = np.minimum((ranks_sel*10)//max(len(tiles), 1), 9)
    sel = []
    for dec in range(10):
        pool = np.where(decile == dec)[0]
        if len(pool) == 0: continue
        sel.append(rng.choice(pool, size=max(1, max_tiles//10), replace=False))
    sel = np.concatenate(sel) if sel else np.array([], dtype=np.int64)
    out = []
    for idx in sel:
        ti, tj, s, e = tiles[idx]; gids = fi[s:e]
        if len(gids) < 1: continue
        ox = tj*TILE; oy = ti*TILE
        xs = ox + np.arange(TILE) + 0.5; ys = oy + np.arange(TILE) + 0.5
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        px = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
        cx0, cy0 = px.mean(0); px[:, 0] -= cx0; px[:, 1] -= cy0
        out.append(dict(px=px, gids=gids, means2d=means2d, conics=conics, opac=opac,
                        colors=colors, M=int(e - s), ti=int(ti), tj=int(tj)))
    return out
