#!/usr/bin/env python3
"""VJS0 probe: linearization-operator feasibility on EPIC-05 (mechanism + timing)."""
import os, sys, time, math
import torch

sys.path.insert(0, "/root/epic05_n1")
import n1_core as C

torch.manual_seed(0)
CELL = os.environ.get("VJS0_CELL", "room@10000")
scene, it = CELL.split("@"); it = int(it)

def P(*a):
    print(" ".join(str(x) for x in a))

ident = C.bootstrap_c0()
P("[boot] c0_ext=%s core=%s" % (ident["so_sha256"][:12], ident["core_sha256"][:12]))

ckpt = C.load_checkpoint(scene, it)
model = C.build_model(ckpt)
C.scene_reset()
master = C.StateMaster(model)
ds = C.load_scene_dataset(scene)
n = len(ds)
holdout = [i for i in range(n) if i % 8 == 0]
train = [i for i in range(n) if i % 8 != 0]

def stratified(idx, k):
    N = len(idx); used=set(); out=[]
    for i in range(k):
        p = int(math.floor((i + 0.5) * N / k))
        while p in used or p >= N: p += 1
        used.add(p); out.append(p)
    return out
src_cams = [train[p] for p in stratified(train, 8)]
P(f"[cams] n={n} train={len(train)} holdout={len(holdout)} src_cams={src_cams}")

fused = C.make_fused_loss()

def render_params(xyz, scaling, rotation, opacity, shs, cam, sh_degree):
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    scaling_a = torch.exp(scaling)
    rotation_a = torch.nn.functional.normalize(rotation, dim=-1)
    opacity_a = torch.sigmoid(opacity).squeeze(-1)
    n_active = (sh_degree + 1) ** 2
    feats = shs[:, :n_active, :].contiguous()
    out = rasterize_gaussian_higs_dynamic(
        xyz, rotation_a, scaling_a, opacity_a, feats,
        sh_degree=sh_degree,
        viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB", eps2d=C.EPS2D)
    return out["frame"].reshape(cam.image_height, cam.image_width, 3).clamp(0, 1)

def exact_fb(model, cam, gt, fused, shd):
    model.optimizer.zero_grad(set_to_none=True)
    img, _, _ = C.render_view(model, cam, shd)
    fused(img, gt).backward()

ci = src_cams[0]
cam, gt = ds.get_item(ci)
P(f"[cam] src {ci} W={cam.image_width} H={cam.image_height}")

xyz0 = model.get_xyz.detach().clone()
scaling0 = model._scaling.detach().clone()
rotation0 = model._rotation.detach().clone()
opacity0 = model._opacity.detach().clone()
shs0 = model._shs.detach().clone()
shd = model.active_sh_degree
N = xyz0.shape[0]
P(f"[model] N={N} active_sh={shd}")

# representative native Adam grad-norm (0.25x reference)
master.restore(); model.optimizer.zero_grad(set_to_none=True)
img0, _, _ = C.render_view(model, cam, shd)
fused(img0, gt).backward()
gn = math.sqrt(sum(float(getattr(model, nm).grad.norm())**2
                   for nm in ("_xyz","_scaling","_rotation","_opacity","_shs")
                   if getattr(model, nm).grad is not None))
P(f"[adam] |grad|_2 = {gn:.6e}")

def make_dirs(scale, seed=1234):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ts = [xyz0, model._scaling, model._rotation, model._opacity, model._shs]
    dirs = [torch.randn(t.shape, generator=g, device="cuda") for t in ts]
    tot = math.sqrt(sum(float(d.norm())**2 for d in dirs))
    return [d * (scale / (tot + 1e-30)) for d in dirs]

scale = 0.25 * gn
dirs = make_dirs(scale)
realized = math.sqrt(sum(float(d.norm())**2 for d in dirs))
P(f"[dir] target norm={scale:.6e} realized={realized:.6e}")

h = 1e-3
t = time.perf_counter()
with torch.no_grad():
    rp = render_params(xyz0+h*dirs[0], scaling0+h*dirs[1], rotation0+h*dirs[2],
                       opacity0+h*dirs[3], shs0+h*dirs[4], cam, shd)
    rm = render_params(xyz0-h*dirs[0], scaling0-h*dirs[1], rotation0-h*dirs[2],
                       opacity0-h*dirs[3], shs0-h*dirs[4], cam, shd)
fd = (rp - rm) / (2*h)
t_fd = time.perf_counter() - t
P(f"[fd] {t_fd:.3f}s |fd|={float(fd.norm()):.6e}")

jvp_ok=False; rel=None; t_jvp=None
try:
    from torch.func import jvp
    primals = (xyz0, scaling0, rotation0, opacity0, shs0)
    def fn(a,b,c,d,e): return render_params(a,b,c,d,e,cam,shd)
    t=time.perf_counter(); out, jd = jvp(fn, primals, tuple(dirs)); t_jvp=time.perf_counter()-t
    rel = float((jd-fd).norm())/(float(fd.norm())+1e-30)
    P(f"[jvp] OK {t_jvp:.3f}s |J.d|={float(jd.norm()):.6e} rel_err_vs_fd={rel:.3e}")
    jvp_ok=True
except Exception as ex:
    P(f"[jvp] FAILED {type(ex).__name__}: {ex}")

def timeit(fn, iters=5, warmup=1):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/iters

t_exact = timeit(lambda: exact_fb(model, cam, gt, fused, shd))
P(f"[time] exact render+fused+backward = {t_exact*1000:.2f} ms")

res = {"cell": CELL, "N": int(N), "n_cams": int(n), "src_cams": src_cams,
       "jvp_ok": jvp_ok, "adam_grad_norm": gn, "delta_norm": scale,
       "fd_s": t_fd, "jvp_s": t_jvp, "rel_err_jvp_vs_fd": rel,
       "exact_fb_ms": t_exact*1000}
C.json_dump(res, "/root/epic05_vjs0/probe_lin.json")
print("PROBE_DONE")
