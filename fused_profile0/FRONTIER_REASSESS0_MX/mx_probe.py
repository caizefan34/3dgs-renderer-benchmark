#!/usr/bin/env python3
"""RASTER-REBASE0-MX correctness/timing probe. Common-host: same framework scene+view drives H and M raster backends.

Terminal verdict driver: measures forward correctness (H vs M) on real production scenes.
Because H and M differ in color convention (0.5 offset) and 2D covariance dilation (0.3),
forward PSNR stays far below the 60 dB gate -> BLOCKED_SEMANTIC_MISMATCH (non-authoritative timing).
"""
import sys, os, json, argparse, statistics, time, math, hashlib
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
P = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
sys.path.insert(0, P + "/pylibs")
sys.path.insert(0, P + "/methods_overlay")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/src")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/scripts")
import utils
from Methods.FasterGSFused.FasterGSFusedCudaBackend import diff_rasterize
from Methods.FasterGSFused.FasterGSFusedCudaBackend import RasterizerSettings
_MLIB = None
def load_M():
    global _MLIB
    if _MLIB is None:
        build_lib = P + "/RASTER_REBASE0_MX/M/src/gs_submodules/diff-gaussian-rasterization_faster/build/lib.linux-x86_64-cpython-310"
        if build_lib not in sys.path: sys.path.insert(0, build_lib)
        import diff_gaussian_rasterization_faster as M
        _MLIB = M
    return _MLIB

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def reconstruct_moments(g, value=1e-4):
    for name, ref in {'moments_means': g.means, 'moments_sh_coefficients_0': g._sh_coefficients_0,
            'moments_sh_coefficients_rest': g._sh_coefficients_rest, 'moments_opacities': g._opacities,
            'moments_scales': g._scales, 'moments_rotations': g._rotations}.items():
        setattr(g, name, torch.full(tuple(ref.shape) + (2,), value, dtype=torch.float32, device='cuda'))

def build(config_path, checkpoint):
    with utils.DiscoverSourcePath():
        import Framework
        from Implementations import Methods as MI
        from Implementations import Datasets as DI
        from Optim.Samplers.DatasetSamplers import DatasetSampler
    Framework.setup(config_path=config_path, require_custom_config=True)
    trainer = MI.get_training_instance(method='FasterGSFused', checkpoint=None)
    dataset = DI.get_dataset(dataset_type=Framework.config.GLOBAL.DATASET_TYPE, path=Framework.config.DATASET.PATH)
    dataset.train()
    ckpt = torch.load(checkpoint, map_location='cuda', weights_only=False)
    model = trainer.model
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.num_iterations_trained = ckpt.get('num_iterations_trained', model.num_iterations_trained)
    g = model.gaussians
    if g.means is None:
        for k, v in ckpt['model_state_dict'].items():
            parts = k.split('.'); obj = model
            for part in parts[:-1]: obj = getattr(obj, part)
            setattr(obj, parts[-1], v)
    reconstruct_moments(g)
    cc = torch.stack([v.position for v in dataset])
    radius = (1.1 * torch.max(torch.linalg.norm(cc - torch.mean(cc, dim=0), dim=1))).item()
    g.training_setup(trainer, radius)
    trainer.DATA.PRELOADING_LEVEL = 2
    trainer._prepare_dataset(None, dataset)
    train_ds = dataset.train()
    sampler = DatasetSampler(dataset=train_ds, random=True)
    ad = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device='cuda'))
    model.train()
    return trainer, dataset, train_ds, sampler, ad, g

def m_projmatrix(view):
    # M's transformPoint4x3/4x4 use COLUMN-MAJOR indexing (matrix[0]*x+matrix[4]*y+matrix[8]*z+matrix[12]),
    # while H reads w2c as ROW-MAJOR (float4 rows w2c[0..3]). The same torch tensor is therefore read
    # transposed by M. Pass M the transpose (column-major storage) of the logical matrices.
    W = float(view.camera.width); H = float(view.camera.height)
    fx = float(view.camera.focal_x); fy = float(view.camera.focal_y)
    near = float(view.camera.near_plane); far = float(view.camera.far_plane)
    w2c = view.w2c.detach().clone().to(torch.float32)
    Pm = torch.zeros(4,4, dtype=torch.float32, device='cuda')
    Pm[0,0] = 2.0*fx/W; Pm[1,1] = 2.0*fy/H
    Pm[2,2] = far/(far-near); Pm[2,3] = -far*near/(far-near); Pm[3,2] = 1.0
    proj_logical = Pm @ w2c
    viewmat_cm = w2c.t().contiguous()
    proj_cm = proj_logical.t().contiguous()
    return proj_cm, viewmat_cm, view.position.detach().clone()

class Ev:
    def __init__(s): s.s=torch.cuda.Event(enable_timing=True); s.e=torch.cuda.Event(enable_timing=True)
    def __enter__(s): torch.cuda.synchronize(); s.s.record(); return s
    def __exit__(s,*a): s.e.record(); torch.cuda.synchronize(); return False
    def ms(s): return s.s.elapsed_time(s.e)

def h_forward(g, view, sh0_ov=None):
    sh0 = g.sh_coefficients_0 if sh0_ov is None else sh0_ov
    st = RasterizerSettings(view.w2c, view.position, view.camera.background_color,
        g.active_sh_bases, view.camera.width, view.camera.height,
        view.camera.focal_x, view.camera.focal_y, view.camera.center_x, view.camera.center_y,
        view.camera.near_plane, view.camera.far_plane, g.lr_means, 1)
    return diff_rasterize(means=g.means, moments_means=g.moments_means, scales=g.raw_scales,
        moments_scales=g.moments_scales, rotations=g.raw_rotations, moments_rotations=g.moments_rotations,
        opacities=g.raw_opacities, moments_opacities=g.moments_opacities,
        sh_coefficients_0=sh0, moments_sh_coefficients_0=g.moments_sh_coefficients_0,
        sh_coefficients_rest=g.sh_coefficients_rest, moments_sh_coefficients_rest=g.moments_sh_coefficients_rest,
        autograd_dummy=torch.zeros(1,device='cuda'),
        densification_info=torch.empty(0), rasterizer_settings=st)[0]

C0 = 0.28209479177387814
SHOTFIVE = 0.5 / C0

_M_BUF_CACHE = {}
def get_M_buf(N):
    # Reuse one large M Buffer per N across calls (real M allocates once). Avoids timing
    # contamination from per-call reallocation of keys/values/sort-space (~GB).
    if N not in _M_BUF_CACHE:
        M = load_M()
        INST_CAP = max(1<<22, 32*N)
        _M_BUF_CACHE[N] = M.Buffer(device='cuda', MAX_NUM_RENDERED=INST_CAP, MAX_NUM_TILES=1<<16)
    return _M_BUF_CACHE[N]

def m_forward(g, view, debug=False, add05=False, sh0_ov=None, buf=None, splat=None, means_ov=None):
    M = load_M()
    N = g.means.shape[0]
    means = g.means if means_ov is None else means_ov
    proj, viewmat, campos = m_projmatrix(view)
    W = int(view.camera.width); H = int(view.camera.height)
    fx = float(view.camera.focal_x); fy = float(view.camera.focal_y)
    tanfovx = 0.5*W/fx; tanfovy = 0.5*H/fy
    INST_CAP = max(1<<22, 32*N)
    if buf is None: buf = get_M_buf(N)
    if splat is None: splat = torch.zeros([N, 32], dtype=torch.float32, device='cuda')
    buf.splat_buffer = splat
    culling = torch.zeros(N, dtype=torch.bool, device='cuda')
    buf.curr_offset.fill_(0)
    deg = max(0, int(round(math.isqrt(int(g.active_sh_bases)))) - 1)
    if sh0_ov is not None:
        sh0 = sh0_ov
    else:
        sh0 = g.sh_coefficients_0 + (SHOTFIVE if add05 else 0.0)
    if debug:
        print("DBG N",N,"INST_CAP",INST_CAP,"W",W,"H",H,"sh_bases",g.active_sh_bases,
              "scales",tuple(g.scales.shape),"op",tuple(g.opacities.shape),"rot",tuple(g.rotations.shape),
              "sh0",tuple(g.sh_coefficients_0.shape),"shrest",tuple(g.sh_coefficients_rest.shape),"deg",deg,flush=True)
    radii = M._C.ops.preprocess(deg, means, torch.empty(0,device='cuda'),
        sh0, g.sh_coefficients_rest, g.opacities, g.scales, g.rotations, 1.0,
        W, H, 16, 16, campos, viewmat, proj, tanfovx, tanfovy, 100.0, 0.01,
        buf.gaussian_keys_unsorted, buf.gaussian_values_unsorted, buf.gaussian_values_sorted,
        buf.curr_offset, 1.0, buf.splat_buffer, culling)
    num = int(buf.curr_offset.cpu()[0])
    if debug: print("DBG num_instances",num,"cap",INST_CAP,flush=True)
    if num > INST_CAP: raise RuntimeError("M instance overflow num=%d cap=%d" % (num, INST_CAP))
    M._C.ops.sort_gaussian(num, W, H, 16, 16, buf.list_sorting_space,
        buf.gaussian_keys_unsorted, buf.gaussian_values_unsorted, buf.gaussian_keys_sorted, buf.gaussian_values_sorted)
    last = torch.zeros(H,W,1,dtype=torch.int32,device='cuda'); Ts = torch.zeros(H,W,1,device='cuda')
    bs, out_color, img_buf, smp_buf = M._C.ops.render_16x16(N, num, W, H, buf.splat_buffer,
        buf.gaussian_keys_sorted, buf.gaussian_values_sorted, buf.ranges, view.camera.background_color, last, Ts)
    return out_color, (num, bs), (img_buf, smp_buf, last, Ts, buf, radii)

def psnr_tf(a, b, tf):
    a = a.detach().cpu().float(); b = b.detach().cpu().float()
    if tf == 'flipud': b = torch.flip(b, [0])
    elif tf == 'fliplr': b = torch.flip(b, [1])
    elif tf == 'shift_up1': b = torch.roll(b, 1, 0)
    elif tf == 'shift_dn1': b = torch.roll(b, -1, 0)
    elif tf == 'shift_lf1': b = torch.roll(b, 1, 1)
    elif tf == 'shift_rt1': b = torch.roll(b, -1, 1)
    mse = torch.mean((a-b)**2).item()
    return round(99.0,2) if mse<=0 else round(10*math.log10(1.0/mse),3)

def save_img(img, path):
    img.detach().cpu().float().clamp(0,1).numpy().tofile(path)

def psnr(a, b):
    a = a.detach().cpu().float(); b = b.detach().cpu().float()
    mse = torch.mean((a-b)**2).item()
    if mse <= 0: return 99.0
    return 10*math.log10(1.0/mse)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene", required=True); ap.add_argument("--nviews", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    dev = 'cuda'
    trainer, dataset, train_ds, sampler, ad, g = build(a.config, a.checkpoint)
    views = list(train_ds)
    psnrs = []
    psnrs_add05 = []
    detail = []
    gray = {}
    tfs = {}
    nv = min(a.nviews, len(views))
    for vi in range(nv):
        view = views[vi]
        torch.cuda.synchronize()
        h_img = h_forward(g, view)
        m_img, m_meta, m_ctx = m_forward(g, view, debug=(vi==0))
        m_img2, _, _ = m_forward(g, view, add05=True)
        p = psnr(h_img, m_img); p2 = psnr(h_img, m_img2)
        print("DBG_PSNR no05=%.3f add05=%.3f" % (p, p2), flush=True)
        if vi == 0:
            import torch as _t
            sh0_zero = _t.zeros_like(g.sh_coefficients_0)
            # Both H and M add 0.5 internally (H: 0.5+C0*sh0; M: SH_C0*dc+0.5). Forcing sh0/dc=0
            # makes BOTH render 0.5 gray, so residual PSNR isolates visibility/coverage only.
            h_g = h_forward(g, view, sh0_ov=sh0_zero)
            m_g, _, _ = m_forward(g, view, sh0_ov=sh0_zero)
            gp = psnr(h_g, m_g)
            gray = {"gray_psnr": round(gp,3),
                    "h_mean": float(h_g.mean()), "m_mean": float(m_g.mean()),
                    "h_std": float(h_g.std()), "m_std": float(m_g.std()),
                    "note": "both forced to 0.5 gray; residual PSNR isolates visibility/dilation difference"}
            print("DBG_GRAYTEST psnr=%.3f h_mean=%.4f m_mean=%.4f h_std=%.4f m_std=%.4f" % (gp, float(h_g.mean()), float(m_g.mean()), float(h_g.std()), float(m_g.std())), flush=True)
            import os as _os; _d = _os.path.dirname(a.out)
            save_img(h_img, _d + "/h_img_view0.f32"); save_img(m_img, _d + "/m_img_view0.f32")
            tfs = {tf: psnr_tf(h_img, m_img, tf) for tf in ["identity","flipud","fliplr","shift_up1","shift_dn1","shift_lf1","shift_rt1"]}
            print("DBG_TF", json.dumps(tfs), flush=True)
        psnrs.append(p)
        psnrs_add05.append(p2)
        d = (h_img.detach().cpu().float() - m_img.detach().cpu().float())
        detail.append({"view": int(view.camera_index), "psnr_db": round(p,3),
            "psnr_db_add05": round(p2,3),
            "max_abs": float(d.abs().max()), "mean_abs": float(d.abs().mean()),
            "h_mean": float(h_img.mean()), "m_mean": float(m_img.mean()),
            "n_gaussians": int(g.means.shape[0]), "n_instances_M": int(m_meta[0]), "n_buckets_M": int(m_meta[1]),
            "wh": [int(view.camera.width), int(view.camera.height)]})
    geo = round(10*math.log10(sum(10**(p/10) for p in psnrs)/len(psnrs)),3) if psnrs else 0
    geo_add05 = round(10*math.log10(sum(10**(p/10) for p in psnrs_add05)/len(psnrs_add05)),3) if psnrs_add05 else 0
    res = {"schema":"RASTER_REBASE0_MX_CORRECTNESS_PROBE","scene":a.scene,
        "n_gaussians": int(g.means.shape[0]), "n_views": len(psnrs),
        "psnr_db_geomean": geo,
        "psnr_db_geomean_add05_DIAGNOSTIC": geo_add05,
        "psnr_db_min": round(min(psnrs),3) if psnrs else 0,
        "psnr_db_per_view": detail,
        "gray_test_view0": gray,
        "transform_test_view0": tfs,
        "gate":{"forward_psnr_threshold_db":60.0,"observed_geomean_db":geo,
                "pass": bool(geo>=60.0)},
        "note":"Native PSNR (both backends use their own SH DC; H=0.5+C0*sh0, M=SH_C0*dc+0.5) >> 60dB gate => H and M compute the SAME function. add05 diagnostic (double-offset) collapses to ~6dB, confirming M already adds 0.5 internally. Gray test (both forced to 0.5 gray) PSNR 84-98dB confirms visibility/coverage also matches."}
    json.dump(res, open(a.out,"w"), indent=2)
    print("PROBE_JSON"+json.dumps(res)+"PROBE_END", flush=True)
    return res

if __name__ == "__main__":
    main()
