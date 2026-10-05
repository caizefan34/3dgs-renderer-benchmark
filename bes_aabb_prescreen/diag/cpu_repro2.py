#!/usr/bin/env python3
"""CPU repro + fix verification for AABB-reject exactness."""
import sys, os, json, math
import numpy as np
os.environ.setdefault("CUDA_HOME", "/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118")
os.environ.setdefault("BES0_PACKAGE_ROOT", "/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH")
sys.path.insert(0, os.environ["BES0_PACKAGE_ROOT"] + "/scripts")
import torch
import scene_adapter as sa
torch.set_num_threads(4)
TS = 16; CULL = 0.01; CAP = 0.999; CUT = 1.0 / 255.0
def build_tile_lists(mu2d, conic, depth, radii, W, H, r_cap):
    mu2d = mu2d.cpu(); conic = conic.cpu(); depth = depth.cpu(); radii = radii.cpu()
    V = mu2d.shape[0]
    tiles_x = (W + TS - 1) // TS; tiles_y = (H + TS - 1) // TS; num_tiles = tiles_x * tiles_y
    r = torch.clamp(radii, max=r_cap)
    tx0 = torch.clamp(torch.floor((mu2d[:, 0] - r) / TS).long(), 0, tiles_x - 1)
    tx1 = torch.clamp(torch.floor((mu2d[:, 0] + r) / TS).long(), 0, tiles_x - 1)
    ty0 = torch.clamp(torch.floor((mu2d[:, 1] - r) / TS).long(), 0, tiles_y - 1)
    ty1 = torch.clamp(torch.floor((mu2d[:, 1] + r) / TS).long(), 0, tiles_y - 1)
    w = (tx1 - tx0 + 1).clamp(min=1); h = (ty1 - ty0 + 1).clamp(min=1)
    counts = w * h; total = int(counts.sum()); seg_start = torch.cumsum(counts, 0) - counts
    ar = torch.arange(total); seg_rep = torch.repeat_interleave(seg_start, counts); local = ar - seg_rep
    gids = torch.repeat_interleave(torch.arange(V), counts)
    w_rep = torch.repeat_interleave(w, counts); tx0_rep = torch.repeat_interleave(tx0, counts); ty0_rep = torch.repeat_interleave(ty0, counts)
    local_tx = local % w_rep; local_ty = local // w_rep
    tx = tx0_rep + local_tx; ty = ty0_rep + local_ty; tile_id = ty * tiles_x + tx
    depth_rep = torch.repeat_interleave(depth, counts)
    dmin = depth_rep.min(); drange = (depth_rep.max() - dmin).clamp(min=1e-6); dn = (depth_rep - dmin) / drange
    composite = tile_id.double() + dn; order = torch.argsort(composite, stable=True)
    return gids[order].numpy(), tile_id[order].numpy(), num_tiles, tiles_x, tiles_y
def eval_colors(scene, cam):
    Rcw = cam.cam_from_world[:3, :3]; tcw = cam.cam_from_world[:3, 3]
    cam_center = -Rcw.t() @ tcw; dirs = scene.xyz - cam_center.unsqueeze(0)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return sa.eval_sh(scene.shs, dirs, scene.active_sh_degree).clamp(min=0.0)
def compute_abba(ca, cb, cc, thr, fix):
    op_pos = np.maximum(thr, 0.0)
    det64 = ca.astype(np.float64) * cc.astype(np.float64) - cb.astype(np.float64) ** 2
    det64 = np.maximum(det64, 1e-30)
    gdx64 = np.sqrt(op_pos.astype(np.float64) * (cc.astype(np.float64) / det64))
    gdy64 = np.sqrt(op_pos.astype(np.float64) * (ca.astype(np.float64) / det64))
    det32 = ca * cc - cb * cb
    safe32 = np.isfinite(det32) & (det32 > 0)
    det32c = np.where(safe32, det32, np.float32(1e-30))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dxmax32 = np.sqrt(np.maximum(op_pos * (cc / det32c), 0.0))
        dymax32 = np.sqrt(np.maximum(op_pos * (ca / det32c), 0.0))
    LARGE = np.float32(1e9)
    dxmax32 = np.where(safe32 & np.isfinite(dxmax32), dxmax32, LARGE).astype(np.float32)
    dymax32 = np.where(safe32 & np.isfinite(dymax32), dymax32, LARGE).astype(np.float32)
    if fix == "max":
        gdx = np.maximum(gdx64, dxmax32.astype(np.float64)); gdy = np.maximum(gdy64, dymax32.astype(np.float64))
        gdx = (gdx * (1 + 1e-6) + 1e-3).astype(np.float32); gdy = (gdy * (1 + 1e-6) + 1e-3).astype(np.float32)
    elif fix == "f32":
        gdx = (dxmax32.astype(np.float64) * (1 + 1e-6) + 1e-3).astype(np.float32)
        gdy = (dymax32.astype(np.float64) * (1 + 1e-6) + 1e-3).astype(np.float32)
    else:  # orig
        gdx = (gdx64 * (1 + 1e-6) + 1e-3).astype(np.float32); gdy = (gdy64 * (1 + 1e-6) + 1e-3).astype(np.float32)
    return gdx, gdy
def walk(g, mx, my, ca, cb, cc, opn, thr, coln, gdx, gdy, tile, ntx, px, py, reject):
    tx = tile % ntx; ty = tile // ntx
    tx0 = float(tx * TS); tx1 = float(tx * TS + TS - 1); ty0 = float(ty * TS); ty1 = float(ty * TS + TS - 1)
    dx = px - mx[g]; dy = py - my[g]
    q = ca[g] * dx * dx + 2.0 * cb[g] * dx * dy + cc[g] * dy * dy
    with np.errstate(over="ignore"):
        e = np.exp(np.clip(-0.5 * q, -1e30, 88.0))
    alpha = np.where(q > thr[g] + CULL, np.float32(0.0), opn[g] * e.astype(np.float32))
    alpha = np.where(alpha < CUT, np.float32(0.0), alpha)
    alpha = np.where(alpha > CAP, np.float32(CAP), alpha).astype(np.float32)
    if reject:
        keep = (mx[g] + gdx[g] >= tx0) & (mx[g] - gdx[g] <= tx1) & (my[g] + gdy[g] >= ty0) & (my[g] - gdy[g] <= ty1)
        alpha = np.where(keep, alpha, np.float32(0.0))
    Cr = np.float32(0); Cg = np.float32(0); Cb = np.float32(0); T = np.float32(1)
    for k in range(len(g)):
        a = alpha[k]
        if a > 0:
            w = T * a
            Cr = np.float32(Cr + w * coln[g[k], 0]); Cg = np.float32(Cg + w * coln[g[k], 1]); Cb = np.float32(Cb + w * coln[g[k], 2])
            T = np.float32(T * (np.float32(1) - a))
    return Cr, Cg, Cb
def main():
    man = json.load(open(os.environ["BES0_PACKAGE_ROOT"] + "/config/scene_manifest.json"))
    rec = man["scenes"]["room"]
    scene = sa.load_scene(rec, "cpu"); cam = scene.cameras[1]
    mu2d, conic, depth, radii = sa.project_gaussians_2d(scene, cam)
    op = sa.opacity_activation(scene.opacity).reshape(-1)
    valid = (depth > 0.05) & (radii > 0); gidx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    mu2d = mu2d[gidx]; conic = conic[gidx]; depth = depth[gidx]; radii = radii[gidx]; op = op[gidx]
    col = eval_colors(scene, cam)[gidx]; W, H = cam.width, cam.height
    ca = conic[:, 0].cpu().numpy().astype(np.float32); cb = conic[:, 1].cpu().numpy().astype(np.float32); cc = conic[:, 2].cpu().numpy().astype(np.float32)
    mx = mu2d[:, 0].cpu().numpy().astype(np.float32); my = mu2d[:, 1].cpu().numpy().astype(np.float32)
    opn = op.cpu().numpy().astype(np.float32)
    thr = (2.0 * (np.log(np.maximum(opn, 1e-12)) + math.log(255.0))).astype(np.float32)
    coln = col.cpu().numpy().astype(np.float32)
    gids, tile_sorted, num_tiles, ntx, nty = build_tile_lists(mu2d, conic, depth, radii, W, H, 256.0)
    p = 6433521; px = p % W; py = p // W; tile = (py // TS) * ntx + (px // TS)
    g = gids[tile_sorted == tile]
    print("[walk] pixel p=%d (px=%d,py=%d) tile=%d len=%d" % (p, px, py, tile, len(g)), flush=True)
    for fix in ["orig", "f32", "max"]:
        gdx, gdy = compute_abba(ca, cb, cc, thr, fix)
        b = walk(g, mx, my, ca, cb, cc, opn, thr, coln, gdx, gdy, tile, ntx, px, py, False)
        t = walk(g, mx, my, ca, cb, cc, opn, thr, coln, gdx, gdy, tile, ntx, px, py, True)
        d = max(abs(float(b[i]) - float(t[i])) for i in range(3))
        print("[fix=%-4s] mode0=(%.6f,%.6f,%.6f) mode3=(%.6f,%.6f,%.6f) maxdiff=%.3e" % (
            fix, *[float(x) for x in b], *[float(x) for x in t], d), flush=True)
    print("[cpu] done", flush=True)
if __name__ == "__main__":
    main()
