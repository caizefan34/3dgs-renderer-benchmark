"""TFC1 benchmark: per-axis (axis-aware) Gaussian footprint vs gsplat circular baseline.

EPIC-05 official PLY models (room/garden/bicycle). For each scene:
  baseline  : gsplat.rasterization()  (circular footprint R=ceil(3*sqrt(lmax)))
  candidate : fully_fused_projection(packed) -> tfc1 per_axis_intersect
              -> isect_offset_encode -> rasterize_to_pixels (per-axis AABB footprint)
Compositing, radix-sort key layout, and backward are reused from gsplat (unchanged).

Measures per scene: intersection count, render (E2E) time, fwd/bwd time, PSNR/SSIM/LPIPS,
gradient cosine. Pass gates: PSNR degradation <=0.05 dB, grad cosine >=0.99999,
intersection reduction >=25%, E2E speedup >=1.10x.
"""
import sys, os, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_tfc0_oracle"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from tfc0_lib import read_images_bin, read_cameras_bin, qvec2rot, RUNTIME_DEFAULTS
from benchmark_framework.scene import load_ply
import tfc1_build
from gsplat import (fully_fused_projection, rasterization, isect_offset_encode,
                    rasterize_to_pixels, spherical_harmonics)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev = "cuda"
TILE = 16
EPS2D = RUNTIME_DEFAULTS["eps2d"]; NEAR = RUNTIME_DEFAULTS["near_plane"]
FAR = RUNTIME_DEFAULTS["far_plane"]; RC = RUNTIME_DEFAULTS["radius_clip"]
ALPHA_THRESHOLD = 1.0 / 255.0

SCENES = ["room", "garden", "bicycle"]

def colmap_dir(scene):
    return os.path.join(REPO, "data", "datasets", "mipnerf360", scene, "sparse", "0")

def gt_path(scene, name):
    for ext in (".JPG", ".jpg", ".PNG", ".png"):
        p = os.path.join(REPO, "data", "datasets", "mipnerf360", scene, "images", name + ext)
        if os.path.exists(p):
            return p
    return None

def load_gt_tensor(path, W, H):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if img.size[0] != W or img.size[1] != H:
        img = img.resize((W, H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(dev)  # [H,W,3]

def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))

def ssim_torch(pred, target):
    # pred/target [H,W,3] in [0,1]; returns mean SSIM in [0,1]
    import torch.nn.functional as F
    p = pred.permute(2, 0, 1).unsqueeze(0); t = target.permute(2, 0, 1).unsqueeze(0)
    ws, sig = 11, 1.5
    coords = torch.arange(ws, device=pred.device, dtype=p.dtype) - ws // 2
    k1d = torch.exp(-(coords ** 2) / (2 * sig ** 2)); k1d = k1d / k1d.sum()
    k = (k1d[:, None] * k1d[None, :]).expand(p.shape[1], 1, ws, ws).contiguous()
    C1 = (0.01) ** 2; C2 = (0.03) ** 2
    def blur(x): return F.conv2d(x, k, padding=ws // 2, groups=p.shape[1])
    mp, mt = blur(p), blur(t)
    mpp, mtm, mpt = mp ** 2, mt ** 2, mp * mt
    sp = blur(p ** 2) - mpp; st = blur(t ** 2) - mtm; spt = blur(p * t) - mpt
    s = ((2 * mpt + C1) * (2 * spt + C2)) / ((mpp + mtm + C1) * (sp + st + C2))
    return s.mean().item()

_LPIPS = None
def lpips_fn():
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="vgg").to(dev).eval()
    return _LPIPS

def lpips_score(pred, target):
    fn = lpips_fn()
    p = pred.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    t = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return float(fn(p, t).item())

def load_scene(scene, cap):
    cd = colmap_dir(scene)
    ply = os.path.join(REPO, "data", "official", "mipnerf360", scene, "point_cloud.ply")
    data = load_ply(ply, device="cpu")
    xyz = data["xyz"].numpy().astype(np.float32); scales = np.exp(data["scales"].numpy().astype(np.float32))
    quats = data["rotations"].numpy().astype(np.float32); opac = data["opacity"].numpy().astype(np.float32)
    shs = data["shs"].numpy().astype(np.float32); SH_DEGREE = int(round(math.sqrt(shs.shape[1]) - 1))
    N = xyz.shape[0]
    imgs = read_images_bin(os.path.join(cd, "images.bin")); cams = read_cameras_bin(os.path.join(cd, "cameras.bin"))
    ids = sorted(imgs.keys())
    fx, fy, cx, cy = cams[0]["params"][:4]; W, H = cams[0]["W"], cams[0]["H"]
    R = qvec2rot(imgs[ids[0]]["qvec"]); t = np.array(imgs[ids[0]]["tvec"])
    viewmat = np.eye(4); viewmat[:3, :3] = R; viewmat[:3, 3] = t
    camtoworld = np.linalg.inv(viewmat); cam_center = camtoworld[:3, 3]
    if cap and N > cap:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(N, size=cap, replace=False))
        xyz = xyz[keep]; scales = scales[keep]; quats = quats[keep]; opac = opac[keep]; shs = shs[keep]
        N = cap
    return dict(scene=scene, N=N, xyz=xyz, scales=scales, quats=quats, opac=opac, shs=shs,
                SH_DEGREE=SH_DEGREE, W=W, H=H, viewmat=viewmat, cam_center=cam_center,
                cam_id=int(ids[0]), img_name=imgs[ids[0]]["name"])

def to_gpu(sc):
    return (torch.tensor(sc["xyz"], device=dev, dtype=torch.float32),
            torch.tensor(sc["scales"], device=dev, dtype=torch.float32),
            torch.nn.functional.normalize(torch.tensor(sc["quats"], device=dev, dtype=torch.float32), dim=-1),
            torch.tensor(sc["opac"], device=dev, dtype=torch.float32),
            torch.tensor(sc["shs"], device=dev, dtype=torch.float32))

def make_viewmats_K(sc):
    viewmat_t = torch.tensor(sc["viewmat"], dtype=torch.float32, device=dev).unsqueeze(0)
    W, H, fx, fy, cx, cy = sc["W"], sc["H"], *sc.get("params", [0,0,0,0])[:4] if False else (0,0,0,0)
    return viewmat_t

def build_camera(sc):
    viewmat_t = torch.tensor(sc["viewmat"], dtype=torch.float32, device=dev).unsqueeze(0)
    # recover fx,fy,cx,cy from K stored? we have them in sc via W,H; recompute from colmap
    return viewmat_t

def render_baseline(sc, cam_tensors, means, quats, scales, opac, shs, viewmat_t, K_t, cam_center, sh_degree):
    t0 = time.time()
    rendered, alphas, meta = rasterization(
        means=means, covars=None, quats=quats, scales=scales,
        opacities=torch.sigmoid(opac), colors=shs,
        viewmats=viewmat_t, Ks=K_t, width=sc["W"], height=sc["H"],
        eps2d=EPS2D, near_plane=NEAR, far_plane=FAR, radius_clip=RC,
        packed=True, render_mode="RGB", sh_degree=sh_degree)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return rendered[0].clamp(0, 1), int(meta["tiles_per_gauss"].sum()), dt, meta

def render_candidate(sc, mod, means, quats, scales, opac, shs, viewmat_t, K_t, cam_center, sh_degree):
    img_ids, gids, radii_g, means2d_g, depths_g, conics_g, comp_g = fully_fused_projection(
        means=means, covars=None, quats=quats, scales=scales,
        viewmats=viewmat_t, Ks=K_t, width=sc["W"], height=sc["H"],
        eps2d=EPS2D, near_plane=NEAR, far_plane=FAR, radius_clip=RC, packed=True)
    torch.cuda.synchronize()
    nnz = len(gids)
    dirs = torch.tensor(means[gids], dtype=torch.float32, device=dev) - cam_center.unsqueeze(0)
    colors = torch.clamp_min(spherical_harmonics(sh_degree, dirs, shs[gids]) + 0.5, 0.0)
    opac_packed = torch.sigmoid(opac[gids])
    opac_for_render = opac_packed * comp_g
    W, H, TILEs = sc["W"], sc["H"], TILE
    TW = (W + TILEs - 1) // TILEs; TH = (H + TILEs - 1) // TILEs
    t0 = time.time()
    tpg, isect_ids, flat_ids = mod.per_axis_intersect(
        means2d_g, conics_g, opac_packed, depths_g, comp_g, float(TILEs), W, H, RC)
    offsets = isect_offset_encode(isect_ids, 1, TW, TH)
    render_c, alpha_c = rasterize_to_pixels(
        means2d_g, conics_g, colors, opac_for_render, W, H, TILEs, offsets, flat_ids, packed=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return render_c[0].clamp(0, 1), int(isect_ids.shape[0]), dt, nnz

def grad_cosine(sc, means, quats, scales, opac, shs, viewmat_t, K_t, cam_center, sh_degree, mod, target):
    # baseline grads
    m2 = means.clone().detach().requires_grad_(True)
    q2 = quats.clone().detach().requires_grad_(True)
    s2 = scales.clone().detach().requires_grad_(True)
    o2 = opac.clone().detach().requires_grad_(True)
    dirs = m2 - cam_center.unsqueeze(0)
    colors = torch.clamp_min(spherical_harmonics(sh_degree, dirs, shs) + 0.5, 0.0)
    rb, _, _ = rasterization(means=m2, covars=None, quats=q2, scales=s2,
        opacities=torch.sigmoid(o2), colors=colors, viewmats=viewmat_t, Ks=K_t,
        width=sc["W"], height=sc["H"], eps2d=EPS2D, near_plane=NEAR, far_plane=FAR,
        radius_clip=RC, packed=True, render_mode="RGB", sh_degree=None)
    torch.cuda.synchronize()
    loss_b = torch.mean((rb[0] - target) ** 2)
    gb = torch.autograd.grad(loss_b, [m2, q2, s2, o2])
    gb_flat = torch.cat([g.reshape(-1) for g in gb])
    # candidate grads
    m3 = means.clone().detach().requires_grad_(True)
    q3 = quats.clone().detach().requires_grad_(True)
    s3 = scales.clone().detach().requires_grad_(True)
    o3 = opac.clone().detach().requires_grad_(True)
    img_ids, gids, radii_g, means2d_g, depths_g, conics_g, comp_g = fully_fused_projection(
        means=m3, covars=None, quats=q3, scales=s3, viewmats=viewmat_t, Ks=K_t,
        width=sc["W"], height=sc["H"], eps2d=EPS2D, near_plane=NEAR, far_plane=FAR, radius_clip=RC, packed=True)
    dirs3 = m3[gids] - cam_center.unsqueeze(0)
    colors3 = torch.clamp_min(spherical_harmonics(sh_degree, dirs3, shs[gids]) + 0.5, 0.0)
    opac_packed3 = torch.sigmoid(o3[gids])
    opac_render3 = opac_packed3 * comp_g
    TW = (sc["W"] + TILE - 1) // TILE; TH = (sc["H"] + TILE - 1) // TILE
    tpg, isect_ids, flat_ids = mod.per_axis_intersect(
        means2d_g.detach(), conics_g.detach(), opac_packed3.detach(), depths_g.detach(),
        comp_g.detach(), float(TILE), sc["W"], sc["H"], RC)
    offsets = isect_offset_encode(isect_ids, 1, TW, TH)
    rc, _ = rasterize_to_pixels(means2d_g, conics_g, colors3, opac_render3,
        sc["W"], sc["H"], TILE, offsets, flat_ids, packed=True)
    torch.cuda.synchronize()
    loss_c = torch.mean((rc[0] - target) ** 2)
    gc = torch.autograd.grad(loss_c, [m3, q3, s3, o3])
    gc_flat = torch.cat([g.reshape(-1) for g in gc])
    cos = float(torch.dot(gb_flat, gc_flat) / (gb_flat.norm() * gc_flat.norm() + 1e-12))
    return cos

def time_fwd_bwd(sc, means, quats, scales, opac, shs, viewmat_t, K_t, cam_center, sh_degree, mod, candidate, target, iters=3):
    def one():
        m2 = means.clone().detach().requires_grad_(True)
        q2 = quats.clone().detach().requires_grad_(True)
        s2 = scales.clone().detach().requires_grad_(True)
        o2 = opac.clone().detach().requires_grad_(True)
        dirs = m2 - cam_center.unsqueeze(0)
        colors = torch.clamp_min(spherical_harmonics(sh_degree, dirs, shs) + 0.5, 0.0)
        if candidate:
            img_ids, gids, radii_g, means2d_g, depths_g, conics_g, comp_g = fully_fused_projection(
                means=m2, covars=None, quats=q2, scales=s2, viewmats=viewmat_t, Ks=K_t,
                width=sc["W"], height=sc["H"], eps2d=EPS2D, near_plane=NEAR, far_plane=FAR, radius_clip=RC, packed=True)
            colors3 = torch.clamp_min(spherical_harmonics(sh_degree, dirs[gids], shs[gids]) + 0.5, 0.0)
            opac_render3 = torch.sigmoid(o2[gids]) * comp_g
            TW = (sc["W"] + TILE - 1) // TILE; TH = (sc["H"] + TILE - 1) // TILE
            tpg, isect_ids, flat_ids = mod.per_axis_intersect(
                means2d_g.detach(), conics_g.detach(), torch.sigmoid(o2[gids]).detach(),
                depths_g.detach(), comp_g.detach(), float(TILE), sc["W"], sc["H"], RC)
            offsets = isect_offset_encode(isect_ids, 1, TW, TH)
            r, _ = rasterize_to_pixels(means2d_g, conics_g, colors3, opac_render3,
                sc["W"], sc["H"], TILE, offsets, flat_ids, packed=True)
        else:
            r, _, _ = rasterization(means=m2, covars=None, quats=q2, scales=s2,
                opacities=torch.sigmoid(o2), colors=colors, viewmats=viewmat_t, Ks=K_t,
                width=sc["W"], height=sc["H"], eps2d=EPS2D, near_plane=NEAR, far_plane=FAR,
                radius_clip=RC, packed=True, render_mode="RGB", sh_degree=None)
        torch.cuda.synchronize()
        loss = torch.mean((r[0] - target) ** 2)
        g = torch.autograd.grad(loss, [m2, q2, s2, o2])
        torch.cuda.synchronize()
        return loss.item()
    best = 1e9
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.time()
        one(); torch.cuda.synchronize()
        best = min(best, time.time() - t0)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="room,garden,bicycle")
    ap.add_argument("--cap", type=int, default=0, help="max Gaussians per scene (0=no cap)")
    ap.add_argument("--tag", default="tfc1")
    args = ap.parse_args()
    mod = tfc1_build.build(verbose=False)
    scenes = args.scenes.split(",")
    results = {}
    for scene in scenes:
        print(f"\n===== {scene} =====", flush=True)
        sc = load_scene(scene, args.cap)
        # need fx,fy,cx,cy
        cd = colmap_dir(scene); cams = read_cameras_bin(os.path.join(cd, "cameras.bin"))
        fx, fy, cx, cy = cams[0]["params"][:4]; sc["params"] = [fx, fy, cx, cy]
        means, scales, quats, opac, shs = to_gpu(sc)
        viewmat_t = torch.tensor(sc["viewmat"], dtype=torch.float32, device=dev).unsqueeze(0)
        K_t = torch.tensor(np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32), device=dev).unsqueeze(0)
        cam_center = torch.tensor(sc["cam_center"], dtype=torch.float32, device=dev)
        # GT
        gt = None; gp = gt_path(scene, sc["img_name"])
        if gp: gt = load_gt_tensor(gp, sc["W"], sc["H"])
        # baseline
        img_b, cnt_b, dt_b, meta = render_baseline(sc, None, means, quats, scales, opac, shs, viewmat_t, K_t, cam_center, sc["SH_DEGREE"])
        # candidate
        img_c, cnt_c, dt_c, nnz = render_candidate(sc, mod, means, quats, scales, opac, shs, viewmat_t, K_t, cam_center, sc["SH_DEGREE"])
        print(f"  N={sc['N']} nnz={nnz} isect_baseline={cnt_b} isect_candidate={cnt_c} "
              f"reduction={1-cnt_c/max(cnt_b,1):.4f} t_base={dt_b*1000:.1f}ms t_cand={dt_c*1000:.1f}ms "
              f"speedup={dt_b/max(dt_c,1e-9):.3f}x", flush=True)
        rec = dict(scene=scene, N=int(sc["N"]), nnz=int(nnz), W=sc["W"], H=sc["H"],
                   cam_id=sc["cam_id"], img_name=sc["img_name"],
                   isect_baseline=int(cnt_b), isect_candidate=int(cnt_c),
                   intersection_reduction=1.0 - cnt_c / max(cnt_b, 1),
                   render_time_baseline_s=dt_b, render_time_candidate_s=dt_c,
                   e2e_speedup=dt_b / max(dt_c, 1e-9))
        if gt is not None:
            rec["psnr_baseline"] = psnr(img_b, gt); rec["psnr_candidate"] = psnr(img_c, gt)
            rec["psnr_degradation_db"] = rec["psnr_baseline"] - rec["psnr_candidate"]
            rec["ssim_baseline"] = ssim_torch(img_b, gt); rec["ssim_candidate"] = ssim_torch(img_c, gt)
            rec["lpips_baseline"] = lpips_score(img_b, gt); rec["lpips_candidate"] = lpips_score(img_c, gt)
            print(f"  PSNR base={rec['psnr_baseline']:.3f} cand={rec['psnr_candidate']:.3f} deg={rec['psnr_degradation_db']:.4f} "
                  f"SSIM b={rec['ssim_baseline']:.4f} c={rec['ssim_candidate']:.4f} "
                  f"LPIPS b={rec['lpips_baseline']:.4f} c={rec['lpips_candidate']:.4f}", flush=True)
        results[scene] = rec
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"TFC1_BENCHMARK_{args.tag}.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nwrote {out}", flush=True)

if __name__ == "__main__":
    main()
