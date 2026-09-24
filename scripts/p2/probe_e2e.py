#!/usr/bin/env python3
"""Verify core gsplat_cuda so + R2 experimental so coexist; run flat BASE forward + TEST native from SAME F9 state."""
import json, os, importlib.util, math, torch

out = {}
CORE = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
R2 = "/mnt/storage_pool/liaoyuanjun/higs_p2_1a_r2_final_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"

def load_so(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

dev = torch.device('cuda')
# Load immutable core ABI -> torch.ops.gsplat.*
try:
    core = load_so(CORE, "gsplat_cuda")
    out["core_loaded"] = True
    out["core_ops"] = {
        "intersect_tile": hasattr(torch.ops.gsplat, "intersect_tile"),
        "rasterize_to_pixels_3dgs": hasattr(torch.ops.gsplat, "rasterize_to_pixels_3dgs"),
    }
except Exception as e:
    out["core_loaded"] = False
    out["core_err"] = f"{type(e).__name__}: {e}"

# Load R2 experimental so -> torch.ops.experimental.*
try:
    r2 = load_so(R2, "experimental_gaussian_render_inference_scene_cuda")
    out["r2_loaded"] = True
    out["r2_schema"] = str(torch.ops.experimental.higs_native_hierarchy_from_projected.default._schema)
    out["r2_has_producer"] = hasattr(r2, "higs_gatherless_projected_producer")
except Exception as e:
    out["r2_loaded"] = False
    out["r2_err"] = f"{type(e).__name__}: {e}"

# --- end-to-end: produce F9 once, run BASE flat (core) and TEST native (R2) on SAME state ---
if out.get("core_loaded") and out.get("r2_loaded"):
    torch.manual_seed(0)
    N = 2000
    W, H, TS = 512, 384, 16
    visible_ids = torch.arange(N, device=dev, dtype=torch.int64)
    means = (torch.rand(N,3,device=dev)*2-1)
    means[:,2] += 5.0
    quats = torch.randn(N,4,device=dev); quats /= quats.norm(dim=-1,keepdim=True)
    scales = torch.rand(N,3,device=dev)*0.06+0.01
    opac = torch.rand(N,device=dev).float()
    colors = torch.rand(N,16,3,device=dev)*0.1
    vm = torch.eye(4,device=dev); vm=vm[None]
    K = torch.tensor([[[W/2,0,W/2],[0,H/2,H/2],[0,0,1]]],device=dev)
    cam_pos = r2.higs_camera_positions_from_viewmats(vm.contiguous())
    radii, means2d, depths, conics, opac_bc, colors_eval = r2.higs_gatherless_projected_producer(
        visible_ids, means.contiguous(), quats.contiguous(), scales.contiguous(),
        opac.contiguous(), colors.contiguous(), vm.contiguous(), K.contiguous(),
        cam_pos, W, H, 0.3, 0.01, 1e10, 0.0)
    radii_c = radii.reshape(1,1,N,2); means2d_c=means2d.reshape(1,1,N,2)
    depths_c=depths.reshape(1,1,N); conics_c=conics.reshape(1,1,N,3); opac_c=opac_bc.reshape(1,1,N)
    colors_c=colors_eval.reshape(1,1,N,3)
    out["f9"] = {"radii": int((radii[:,0]>0).sum().item()), "conics_dtype": str(conics.dtype)}
    TW = math.ceil(W/TS); TH = math.ceil(H/TS)
    try:
        tiles_per, isect_ids, flatten_ids = torch.ops.gsplat.intersect_tile(
            means2d_c, radii_c, depths_c, conics_c, opac_c, None, None, 1,
            TS, TW, TH, True, False, None)
        out["base_intersect"] = {"n_isects": int(isect_ids.numel())}
        import gsplat.rasterization as _r
        # isect_offset_encode with [C,w,h]
        isect_offsets = torch.zeros((1,1,TH,TW), device=dev, dtype=torch.int32)
        _r.isect_offset_encode_nonexistent  # ensure namespace exists
    except Exception as e:
        out["base_intersect_err"] = f"{type(e).__name__}: {e}"
    torch.cuda.synchronize()
    rgb_t, alpha_t, diag = torch.ops.experimental.higs_native_hierarchy_from_projected(
        visible_ids, radii, means2d, depths, conics, opac_bc, colors_eval, W, H, TS,
        torch.tensor([0.05,0.1,0.15],device=dev), True)
    torch.cuda.synchronize()
    out["test"] = {"rgb": list(rgb_t.shape), "alpha": list(alpha_t.shape),
                   "n_diag": len(diag), "finite": bool(torch.isfinite(rgb_t).all()),
                   "rgb_absmax": float(rgb_t.abs().max().item())}

print(json.dumps(out, sort_keys=True, indent=2))