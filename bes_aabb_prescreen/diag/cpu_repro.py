#!/usr/bin/env python3
"""CPU-only reproduction of the AABB-reject exactness bug. No GPU needed.
Replicates the kernel float32 q evaluation + float64-derived AABB, and
checks whether the float32 ellipse can exceed the AABB (wrong skips)."""
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
    tiles_x = (W + TS - 1) // TS; tiles_y = (H + TS - 1) // TS
    num_tiles = tiles_x * tiles_y
    r = torch.clamp(radii, max=r_cap)
    tx0 = torch.clamp(torch.floor((mu2d[:, 0] - r) / TS).long(), 0, tiles_x - 1)
    tx1 = torch.clamp(torch.floor((mu2d[:, 0] + r) / TS).long(), 0, tiles_x - 1)
    ty0 = torch.clamp(torch.floor((mu2d[:, 1] - r) / TS).long(), 0, tiles_y - 1)
    ty1 = torch.clamp(torch.floor((mu2d[:, 1] + r) / TS).long(), 0, tiles_y - 1)
    w = (tx1 - tx0 + 1).clamp(min=1); h = (ty1 - ty0 + 1).clamp(min=1)
    counts = w * h; total = int(counts.sum())
    seg_start = torch.cumsum(counts, 0) - counts
    ar = torch.arange(total)
    seg_rep = torch.repeat_interleave(seg_start, counts)
    local = ar - seg_rep
    gids = torch.repeat_interleave(torch.arange(V), counts)
    w_rep = torch.repeat_interleave(w, counts)
    tx0_rep = torch.repeat_interleave(tx0, counts); ty0_rep = torch.repeat_interleave(ty0, counts)
    local_tx = local % w_rep; local_ty = local // w_rep
    tx = tx0_rep + local_tx; ty = ty0_rep + local_ty
    tile_id = ty * tiles_x + tx
    depth_rep = torch.repeat_interleave(depth, counts)
    dmin = depth_rep.min(); drange = (depth_rep.max() - dmin).clamp(min=1e-6)
    dn = (depth_rep - dmin) / drange
    composite = tile_id.double() + dn
    order = torch.argsort(composite, stable=True)
    return gids[order].numpy(), tile_id[order].numpy(), num_tiles, tiles_x, tiles_y

def eval_colors(scene, cam):
    Rcw = cam.cam_from_world[:3, :3]; tcw = cam.cam_from_world[:3, 3]
    cam_center = -Rcw.t() @ tcw
    dirs = scene.xyz - cam_center.unsqueeze(0)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    rgb = sa.eval_sh(scene.shs, dirs, scene.active_sh_degree)
    return rgb.clamp(min=0.0)

def main():
    man = json.load(open(os.environ["BES0_PACKAGE_ROOT"] + "/config/scene_manifest.json"))
    rec = man["scenes"]["room"]
    scene = sa.load_scene(rec, "cpu")
    cam = scene.cameras[1]
    mu2d, conic, depth, radii = sa.project_gaussians_2d(scene, cam)
    op = sa.opacity_activation(scene.opacity).reshape(-1)
    valid = (depth > 0.05) & (radii > 0)
    gidx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    mu2d = mu2d[gidx]; conic = conic[gidx]; depth = depth[gidx]; radii = radii[gidx]; op = op[gidx]
    col = eval_colors(scene, cam)[gidx]
    W, H = cam.width, cam.height
    print("[cpu] cam=1 W=%d H=%d N=%d" % (W, H, mu2d.shape[0]), flush=True)
    ca = conic[:, 0].cpu().numpy().astype(np.float32)
    cb = conic[:, 1].cpu().numpy().astype(np.float32)
    cc = conic[:, 2].cpu().numpy().astype(np.float32)
    mx = mu2d[:, 0].cpu().numpy().astype(np.float32)
    my = mu2d[:, 1].cpu().numpy().astype(np.float32)
    opn = op.cpu().numpy().astype(np.float32)
    thr = (2.0 * (np.log(np.maximum(opn, 1e-12)) + math.log(255.0))).astype(np.float32)
    Tpos = np.maximum(thr, 0.0).astype(np.float32)
    # float64 AABB (as probe computes)
    det64 = ca.astype(np.float64) * cc.astype(np.float64) - cb.astype(np.float64) ** 2
    det64 = np.maximum(det64, 1e-30)
    s11_64 = cc.astype(np.float64) / det64
    s22_64 = ca.astype(np.float64) / det64
    gdx64 = np.sqrt(Tpos.astype(np.float64) * s11_64)
    gdy64 = np.sqrt(Tpos.astype(np.float64) * s22_64)
    gdx = (gdx64 * (1 + 1e-6) + 1e-3).astype(np.float32)
    gdy = (gydy64 * (1 + 1e-6) + 1e-3).astype(np.float32) if False else (gdy64 * (1 + 1e-6) + 1e-3).astype(np.float32)
    # float32 ellipse extent (kernel-consistent): s11_f32 = cc/(ca*cc-cb^2) in f32
    det32 = (ca * cc - cb * cb)
    det32c = np.maximum(det32, np.float32(1e-30))
    s11_32 = cc / det32c
    s22_32 = ca / det32c
    dxmax32 = np.sqrt(np.maximum(Tpos * s11_32, 0.0))
    dymax32 = np.sqrt(np.maximum(Tpos * s22_32, 0.0))
    # hypothesis test: how often does float32 ellipse exceed float64 AABB?
    over_x = dxmax32 > gdx64 * (1 + 1e-6) + 1e-3
    over_y = dymax32 > gdy64 * (1 + 1e-6) + 1e-3
    print("[hyp] n_ellipse_wider_x=%d y=%d of %d" % (int(over_x.sum()), int(over_y.sum()), len(ca)), flush=True)
    if over_x.any():
        j = int(np.argmax(np.where(over_x, dxmax32 - gdx64, -1)))
        print("[hyp] worst x: gid=%d dxmax32=%.4f gdx64=%.4f ratio=%.4f ca=%.4e cb=%.4e cc=%.4e det32=%.4e det64=%.4e" % (
            j, dxmax32[j], gdx64[j], dxmax32[j] / max(gdx64[j], 1e-9), ca[j], cb[j], cc[j], det32[j], det64[j]), flush=True)
    # Now reproduce the kernel walk for the worst pixel (p=6433521 -> ch2)
    gids, tile_sorted, num_tiles, ntx, nty = build_tile_lists(mu2d, conic, depth, radii, W, H, 256.0)
    p = 6433521
    px = p % W; py = p // W
    tx = px // TS; ty = py // TS; tile = ty * ntx + tx
    sel = tile_sorted == tile
    g = gids[sel]
    print("[walk] pixel p=%d (px=%d,py=%d) tile=%d tilelist_len=%d" % (p, px, py, tile, len(g)), flush=True)
    coln = col.cpu().numpy().astype(np.float32)
    def walk(reject):
        Cr = np.float32(0); Cg = np.float32(0); Cb = np.float32(0); T = np.float32(1)
        fx = np.float32(px); fy = np.float32(py)
        for k in range(len(g)):
            gid = int(g[k])
            if reject:
                mxx = mx[gid]; myy = my[gid]
                if not (mxx + gdx[gid] >= np.float32(tx * TS) and mxx - gdx[gid] <= np.float32(tx * TS + TS - 1) and
                        myy + gdy[gid] >= np.float32(ty * TS) and myy - gdy[gid] <= np.float32(ty * TS + TS - 1)):
                    continue
            dx = fx - mx[gid]; dy = fy - my[gid]
            q = np.float32(ca[gid] * dx * dx + np.float32(2.0 * cb[gid] * dx * dy) + cc[gid] * dy * dy)
            if q > thr[gid] + CULL:
                continue
            alpha = np.float32(opn[gid] * np.float32(math.exp(-0.5 * float(q))))
            if alpha < CUT:
                continue
            if alpha > CAP:
                alpha = np.float32(CAP)
            w = T * alpha
            Cr = np.float32(Cr + w * coln[gid, 0]); Cg = np.float32(Cg + w * coln[gid, 1]); Cb = np.float32(Cb + w * coln[gid, 2])
            T = np.float32(T * (np.float32(1) - alpha))
        return Cr, Cg, Cb
    b = walk(False); t = walk(True)
    print("[walk] mode0=(%.6f,%.6f,%.6f) mode3=(%.6f,%.6f,%.6f) diff=%.6f" % (
        b[0], b[1], b[2], t[0], t[1], t[2], max(abs(b[i] - t[i]) for i in range(3))), flush=True)
    # find first wrongly-skipped gaussian (skip in reject but alpha>0 in baseline)
    fx = np.float32(px); fy = np.float32(py)
    T = np.float32(1)
    for k in range(len(g)):
        gid = int(g[k])
        dx = fx - mx[gid]; dy = fy - my[gid]
        q = np.float32(ca[gid] * dx * dx + np.float32(2.0 * cb[gid] * dx * dy) + cc[gid] * dy * dy)
        alpha = np.float32(0)
        if not (q > thr[gid] + CULL):
            alpha = np.float32(opn[gid] * np.float32(math.exp(-0.5 * float(q))))
            if alpha < CUT: alpha = np.float32(0)
            elif alpha > CAP: alpha = np.float32(CAP)
        mxx = mx[gid]; myy = my[gid]
        keep = (mxx + gdx[gid] >= np.float32(tx * TS) and mxx - gdx[gid] <= np.float32(tx * TS + TS - 1) and
                myy + gdy[gid] >= np.float32(ty * TS) and myy - gdy[gid] <= np.float32(ty * TS + TS - 1))
        if (not keep) and alpha > 0:
            print("[WRONG SKIP] gid=%d alpha=%.6f q=%.4f T=%.4f  dxmax32=%.4f gdx=%.4f dymax32=%.4f gdy=%.4f ca=%.4e cb=%.4e cc=%.4e det32=%.4e det64=%.4e" % (
                gid, alpha, q, thr[gid], dxmax32[gid], gdx[gid], dymax32[gid], gdy[gid], ca[gid], cb[gid], cc[gid], det32[gid], det64[gid]), flush=True)
            break
    print("[cpu] done", flush=True)

if __name__ == "__main__":
    main()
