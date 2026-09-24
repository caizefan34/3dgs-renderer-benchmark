#!/usr/bin/env python3
"""Verify the exact flat BASE forward recipe on gsplat 1.5.3 core ABI using F9 state."""
import json, torch, importlib.util, math

out = {}
CORE = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
R2 = "/mnt/storage_pool/liaoyuanjun/higs_p2_1a_r2_final_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
def load_so(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
dev = torch.device('cuda')
core = load_so(CORE, "gsplat_cuda")
r2 = load_so(R2, "experimental_gaussian_render_inference_scene_cuda")

# Check ops on core ABI
out["has_isect_offset_encode"] = hasattr(torch.ops.gsplat, "isect_offset_encode")
out["has_rasterize_to_pixels_2dgs"] = hasattr(torch.ops.gsplat, "rasterize_to_pixels_3dgs")

torch.manual_seed(1)
N, W, H, TS = 4000, 640, 480, 16
visible_ids = torch.arange(N, device=dev, dtype=torch.int64)
means = torch.rand(N,3,device=dev)*2-1; means[:,2]+=5.0
quats = torch.randn(N,4,device=dev); quats/=quats.norm(dim=-1,keepdim=True)
scales = torch.rand(N,3,device=dev)*0.06+0.01
opac = torch.rand(N,device=dev).float()
colors = torch.rand(N,16,3,device=dev)*0.1
vm = torch.eye(4,device=dev)[None]
K = torch.tensor([[[W/2,0,W/2],[0,H/2,H/2],[0,0,1]]],device=dev)
cam_pos = r2.higs_camera_positions_from_viewmats(vm.contiguous())
radii, means2d, depths, conics, opac_bc, colors_eval = r2.higs_gatherless_projected_producer(
    visible_ids, means.contiguous(), quats.contiguous(), scales.contiguous(),
    opac.contiguous(), colors.contiguous(), vm.contiguous(), K.contiguous(),
    cam_pos, W, H, 0.3, 0.01, 1e10, 0.0)
radii_c=radii.reshape(1,1,N,2); means2d_c=means2d.reshape(1,1,N,2)
depths_c=depths.reshape(1,1,N); conics_c=conics.reshape(1,1,N,3)
opac_c=opac_bc.reshape(1,1,N); colors_c=colors_eval.reshape(1,1,N,3)
TW=math.ceil(W/TS); TH=math.ceil(H/TS)

# intersect_tile: try signatures
for n_args in (13,):
    try:
        tiles_per, isect_ids, flatten_ids = torch.ops.gsplat.intersect_tile(
            means2d_c, radii_c, depths_c, conics_c, opac_c,
            None, None, 1, TS, TW, TH, True, False)
        out["intersect_tile13"] = {"ok": True, "isects": int(isect_ids.numel())}
        # intersect_offset (frozen core op), returns offsets tensor
        offs = torch.ops.gsplat.intersect_offset(isect_ids, 1, TW, TH)
        out["intersect_offset"] = {"shape": list(offs.shape), "dtype": str(offs.dtype),
                                   "finite": bool(torch.isfinite(offs).all().item())}
        break
    except Exception as e:
        out.setdefault("intersect_errors", {})[str(n_args)] = f"{type(e).__name__}: {str(e)[:200]}"

# rasterize_to_pixels_3dgs signature
try:
    bg = torch.zeros((1,1,3),device=dev)+0.05
    render_out = torch.ops.gsplat.rasterize_to_pixels_3dgs(
        means2d_c, conics_c, colors_c, opac_c, bg, None, W, H, TS,
        offs.reshape(1,TH,TW).contiguous(), flatten_ids.contiguous(), False, False)
    renders, alphas, last_ids = render_out[0], render_out[1], render_out[2]
    out["rasterize"] = {"renders": list(renders.shape), "alphas": list(alphas.shape),
                        "finite": bool(torch.isfinite(renders).all()),
                        "absmax": float(renders.abs().max().item())}
except Exception as e:
    out["rasterize_err"] = f"{type(e).__name__}: {str(e)[:300]}"
print(json.dumps(out, sort_keys=True, indent=2, default=str))