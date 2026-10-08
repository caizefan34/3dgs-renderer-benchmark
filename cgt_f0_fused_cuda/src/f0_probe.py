#!/usr/bin/env python3
"""F0 probe v2: shapes + coordinate-frame test (centered vs global px)."""
import sys, numpy as np, torch, gsplat
sys.path.insert(0, "/home/liaoyuanjun/cgt_realscene/final/SCRIPTS")
from g1_compressibility import read_cameras_bin, read_images_bin, q2R, tile_teacher, POWER_CLIP
DATA = "/tmp/3dgs-renderer-benchmark/data"
scene = "room"
ckpt = "/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_multiphase_v2/checkpoints/iter_15000.pt"
dev = "cuda"
cams = read_cameras_bin(f"{DATA}/datasets/mipnerf360/{scene}/sparse/0/cameras.bin")
imgs = read_images_bin(f"{DATA}/datasets/mipnerf360/{scene}/sparse/0/images.bin")
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
means = ck["xyz"].to(dev).float(); scales = torch.exp(ck["scaling"].to(dev).float())
quats = ck["rotation"].to(dev).float(); opacities = torch.sigmoid(ck["opacity"].to(dev).float()); shs = ck["shs"].to(dev).float()
deg = int(ck["active_sh_degree"]); print("N gaussians", means.shape, "sh_degree", deg)
iid, name, cam, qv, tv = imgs[0]; mid, W, H, par = cams[cam]; fx, fy, cx, cy = par
R_w2c = q2R(qv); t_w2c = np.array(tv); t_c2w = -R_w2c.T @ t_w2c
viewmat = torch.eye(4, device=dev)[None]
viewmat[0, :3, :3] = torch.as_tensor(R_w2c, device=dev).float()
viewmat[0, :3, 3] = torch.as_tensor(t_w2c, device=dev).float()
K = torch.as_tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], device=dev).float()
campos = torch.as_tensor(t_c2w, device=dev).float()[None]
rc, ra, meta = gsplat.rasterization(means, quats, scales, opacities, shs, viewmat, K, width=W, height=H,
                                    sh_degree=deg, render_mode="RGB", packed=True, tile_size=16)
print("meta keys", sorted(meta.keys()))
for k in ["means2d", "conics", "opacities", "depths", "gaussian_ids", "flatten_ids", "isect_offsets"]:
    v = meta[k]
    print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
def flat2(k):
    x = meta[k].detach().cpu().numpy(); return x[0] if x.ndim == 3 else x
means2d = flat2("means2d"); conics = flat2("conics"); opac = flat2("opacities")
gids_full = meta["gaussian_ids"].detach().cpu().numpy()
fi = meta["flatten_ids"].detach().cpu().numpy()
io = meta["isect_offsets"].detach().cpu().numpy().reshape(-1)
th, tw = int(meta["tile_height"]), int(meta["tile_width"])
print("means2d", means2d.shape, "conics", conics.shape, "opac", opac.shape, "gids_full", gids_full.shape, "fi", fi.shape)
print("gids_full range", gids_full.min(), gids_full.max(), "fi range", fi.min(), fi.max())
print("tiles", th, tw, "total_intersections", len(fi))
dirs = torch.nn.functional.normalize(means - campos, dim=-1)
colors = (gsplat.spherical_harmonics(deg, dirs, shs) + 0.5).clamp(0, 1).detach().cpu().numpy()[gids_full]
print("colors", colors.shape, "range", colors.min(), colors.max())
starts = io.astype(np.int64)
for (ti, tj) in [(th//2, tw//2), (5, 5)]:
    s = int(starts[ti*tw+tj]); e = int(starts[ti*tw+tj+1]) if tj+1 < tw else int(starts[(ti+1)*tw])
    gids = fi[s:e]
    ox, oy = tj*16, ti*16
    xs = ox+np.arange(16)+0.5; ys = oy+np.arange(16)+0.5
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    pxg = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
    px = pxg.copy(); c0 = px.mean(0); px[:,0]-=c0[0]; px[:,1]-=c0[1]
    print(f"--- tile({ti},{tj}) M={e-s} ox,oy={ox},{oy} gids[{gids.min()},{gids.max()}]")
    print(f"    means2d[gids] x[{float(means2d[gids][:,0].min()):.1f},{float(means2d[gids][:,0].max()):.1f}] y[{float(means2d[gids][:,1].min()):.1f},{float(means2d[gids][:,1].max()):.1f}]")
    print(f"    conics[gids] sample {conics[gids][:2]}")
    try:
        E, T = tile_teacher(px[:,0:1], px[:,1:2], gids, means2d, conics, opac, colors)
        print(f"    CENTERED-px E[min,max,mean]=({E.min():.4f},{E.max():.4f},{E.mean():.5f}) T=({T.min():.4f},{T.max():.4f})")
    except Exception as ex:
        print("    CENTERED-px FAIL", repr(ex))
    try:
        E2, T2 = tile_teacher(pxg[:,0:1], pxg[:,1:2], gids, means2d, conics, opac, colors)
        print(f"    GLOBAL-px   E[min,max,mean]=({E2.min():.4f},{E2.max():.4f},{E2.mean():.5f}) T=({T2.min():.4f},{T2.max():.4f})")
    except Exception as ex:
        print("    GLOBAL-px FAIL", repr(ex))
print("PROBE2_DONE")
