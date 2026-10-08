import sys, time, math
sys.path.insert(0, r"C:\Users\36570\3dgs-renderer-benchmark\_habc0_exec_pack\HABC0_ADAPTIVE_BACKWARD_COMPRESSION_ORACLE_PACK\oracle_src")
import numpy as np, torch
import habc0_oracle as O

BASE = r"C:\Users\36570\3dgs-renderer-benchmark\data\datasets\mipnerf360"
PLY = BASE + r"\room\point_cloud.ply"
scene = "room"

t0 = time.time()
model, finite = O.load_model(PLY, device="cpu")
print("loaded room n=%d finite_frac=%.4f t=%.1fs" % (model.n, float(finite.float().mean()), time.time()-t0))
keep = finite.clone()
m = O.GSModel(model.means[keep], model.raw_scales[keep], model.raw_rotations[keep],
              model.raw_opacities[keep], model.sh0[keep], model.shrest[keep], int(keep.sum()))
print("kept n=%d" % m.n)

cams = O.read_colmap_cameras(BASE + "\\" + scene + r"\sparse\0\cameras.bin")
imgs = O.read_colmap_images(BASE + "\\" + scene + r"\sparse\0\images.bin")
cam = cams[0]; W, H = cam["w"], cam["h"]; fx, fy, cx, cy = cam["params"]
print("cam0 WxH=%dx%d f=%.1f,%.1f c=%.1f,%.1f n_cams=%d" % (W,H,fx,fy,cx,cy,len(imgs)))

t0 = time.time()
im = imgs[0]
M = O.w2c_from_qvec_tvec(im["qvec"], im["tvec"]); R = M[:,:3]; t = M[:,3]
cam_pos = -R.T @ t
proj, _ = O.project(m, M, cam_pos, W, H, fx, fy, cx, cy)
print("project t=%.1fs visible=%d" % (time.time()-t0, int(proj["visible"].sum())))

t0 = time.time()
tiles = O.build_tiles(proj, W, H, max_tiles=None, seed=0)
tb = tiles["tile_bounds"]; per_tile = np.diff(tb)
print("build_tiles t=%.1fs total_inst=%d n_tiles_grid=%d nonempty=%d" % (
    time.time()-t0, int(per_tile.sum()), tiles["n_tiles"], int((per_tile>0).sum())))

# busiest tile
st = tiles["sampled_tiles"]
busiest = int(st[np.argmax(per_tile[st])])
sg = torch.from_numpy(tiles["sorted_gauss"][tiles["tile_bounds"][busiest]:tiles["tile_bounds"][busiest+1]].astype(np.int64))
print("busiest tile=%d M=%d n_buckets=%d" % (busiest, sg.numel(), math.ceil(sg.numel()/32)))

BG = torch.zeros(3, dtype=torch.float64)
t0 = time.time()
image_tile, saved_states, T_final, n_used = O.blend_tile(sg, proj, busiest, tiles["gwx"], W, H, BG)
print("blend_tile t=%.1fs n_buckets=%d" % (time.time()-t0, len(saved_states)))

Ts = torch.cat([T for (_,T) in saved_states])
RGB = torch.cat([Cp for (Cp,_) in saved_states], dim=0)
print("T range [%.4f, %.4f] p50=%.4f" % (float(Ts.min()), float(Ts.max()), float(Ts.quantile(0.5))))
print("RGB range [%.4f, %.4f] mean=%.4f" % (float(RGB.min()), float(RGB.max()), float(RGB.mean())))
near_zero = float((RGB.abs()<1e-6).float().mean())
sat = float(((RGB>0.999)|(RGB<1e-6)).float().mean())
print("RGB near_zero_frac=%.4f sat_frac=%.4f" % (near_zero, sat))

# backward sanity: replay vs itself
torch_rng = torch.Generator().manual_seed(0)
g = torch.randn(image_tile.shape, dtype=torch.float64, generator=torch_rng)
ref = O.backward_tile(sg, proj, busiest, tiles["gwx"], W, H, BG, g, image_tile, saved_states, T_final, n_used)
print("backward grad shapes:", [tuple(x.shape) for x in ref])
print("opacity grad absmean=%.4e" % float(ref[2].abs().mean()))
print("SMOKE OK total=%.1fs" % (time.time()-t0))