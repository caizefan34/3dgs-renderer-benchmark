#!/usr/bin/env python3
import sys, os, time, math
sys.path.insert(0, "/root/hm0_pkg")
sys.path.insert(0, "/root/epic05_moonshot/a0")
import hm0_common as C
import a0_lib as A
import torch
import torch.nn.functional as F

def sync(): torch.cuda.synchronize()

def isolated_group_render(m, gidx, cams, sh):
    means = m["xyz"][gidx]
    quats = F.normalize(m["rotation"][gidx], dim=-1)
    scales = torch.exp(m["scaling"][gidx])
    opac = torch.sigmoid(m["opacity"][gidx])
    colors = m["shs"][gidx]
    viewmats = torch.stack([c.viewmatrix for c in cams]).unsqueeze(1)  # [C,1,4,4]
    Ks = torch.stack([c.K for c in cams]).unsqueeze(1)                  # [C,1,3,3]
    H, W = cams[0].image_height, cams[0].image_width
    bg = torch.zeros(1, 3, device="cuda")
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    out = rasterize_gaussian_higs_dynamic(
        means, quats, scales, opac, colors, sh_degree=sh,
        viewmats=viewmats, Ks=Ks, width=W, height=H,
        background=bg, render_mode="RGB", enable_culling=True,
        camera_model="pinhole", backward_mode="higs_native", eps2d=0.3)
    return out["frame"], out["alpha"]

for scene, it in (("room", 10000), ("bicycle", 10000)):
    t0 = time.perf_counter()
    env = C.bootstrap_c0()
    ds = C.load_dataset(scene, device="cuda")
    n_cams = len(ds)
    cam_ids = C.frozen_camera_ids(n_cams)
    m, opt_state, ident, osd, slrs = C.load_model(scene, it, device="cuda")
    sh = min(ident["active_sh_degree"], C.SH_DEGREE)
    N = ident["num_points"]
    print(f"[{scene}@{it}] N={N} n_cams={n_cams} sh={sh} load={time.perf_counter()-t0:.1f}s", flush=True)

    # intersection-count pass over 8 train cams
    t1 = time.perf_counter()
    info_uni, uninstall = A.install_capture_patch()
    A.A0_CAPTURE["enabled"] = True
    isect = torch.zeros(N, device="cuda")
    for ci in cam_ids:
        A.A0_CAPTURE["records"].clear()
        cam, gt = ds.get_item(ci)
        frame, info, out = C.render_c0(m, cam, sh)
        loss = C.train_loss(frame, gt, C.LAMBDA_DSSIM)
        loss.backward()
        rec = A.A0_CAPTURE["records"][-1]
        cap = A.parse_capture(rec)
        vis = cap["visible_ids"]                    # compact->global
        flat = cap["flatten_ids_f"]                 # compact row ids per hit
        g = vis[flat.reshape(-1)]                   # global ids
        isect += torch.bincount(g, minlength=N).float()
        optimizer = None
    uninstall()
    sync()
    print(f"[{scene}@{it}] isect pass 8cams={time.perf_counter()-t1:.2f}s total_hits={int(isect.sum())} vis_gaussians={int((isect>0).sum())}", flush=True)

    # isolated group render: K=16 nearest of a visible seed
    seed = int((isect > 0).nonzero()[0])
    t2 = time.perf_counter()
    with torch.no_grad():
        d = torch.cdist(m["xyz"][seed:seed+1], m["xyz"]).reshape(-1)
        nn = torch.topk(d, 17, largest=False).indices[1:17].tolist()
    gidx = torch.tensor(nn, device="cuda")
    cams = [ds.get_camera(ci) for ci in cam_ids]
    # warm
    for _ in range(2):
        fr, al = isolated_group_render(m, gidx, cams, sh); sync()
    t3 = time.perf_counter()
    nR = 20
    for _ in range(nR):
        fr, al = isolated_group_render(m, gidx, cams, sh)
        (fr.sum()+al.sum()).backward(); sync()
    print(f"[{scene}@{it}] group render K=16 8cams fwd+bwd={1000*(time.perf_counter()-t3)/nR:.1f}ms/iter", flush=True)
    torch.cuda.empty_cache()
print("PROBE_DONE", flush=True)