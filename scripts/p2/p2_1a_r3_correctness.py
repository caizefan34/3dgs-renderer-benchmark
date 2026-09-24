#!/usr/bin/env python3
"""R3 hard-stop forward correctness collection.  Deliberately contains no timers."""
import hashlib, importlib.util, json, math, os
import torch

CORE = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
TEST = "/mnt/storage_pool/liaoyuanjun/higs_p2_1a_r3_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CKPT = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7"
CAMS = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360"
OUT = "/mnt/storage_pool/liaoyuanjun/higs_p2_1a_r3_runtime"
MAX_LONG, TS = 2048, 16

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def setup(scene, ext, device):
    cam = json.load(open(f"{CAMS}/{scene}/cameras.json"))[0]
    ck = torch.load(f"{CKPT}/a100_30k_{scene}_t16_16/a100_30k_{scene}_t16_16_latest.pt", map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck); scale = MAX_LONG / max(float(cam["width"]), float(cam["height"]))
    width, height = int(round(cam["width"] * scale)), int(round(cam["height"] * scale))
    rot = torch.tensor(cam["rotation"], dtype=torch.float32, device=device); pos = torch.tensor(cam["position"], dtype=torch.float32, device=device)
    view = torch.eye(4, dtype=torch.float32, device=device); view[:3, :3] = rot; view[:3, 3] = -rot @ pos
    K = torch.tensor([[cam["fx"] * scale, 0, width / 2 - .5], [0, cam["fy"] * scale, height / 2 - .5], [0, 0, 1]], dtype=torch.float32, device=device)[None]
    means = state["xyz"].to(device).contiguous(); n = means.shape[0]; ids = torch.arange(n, device=device, dtype=torch.int64)
    p = ext.higs_gatherless_projected_producer(ids, means, state["rotations"].to(device).contiguous(), torch.exp(state["scales"].to(device)).contiguous(), torch.sigmoid(state["opacity"].to(device).flatten()).contiguous(), state["shs"].to(device).contiguous(), view[None].contiguous(), K.contiguous(), ext.higs_camera_positions_from_viewmats(view[None].contiguous()), width, height, .3, .01, 1e10, 0.)
    return ids, *p, width, height

def stats(ref, got):
    d = (ref - got).float(); out = {"max_abs": float(d.abs().max()), "mean_abs": float(d.abs().mean()), "rel_l2": float(d.norm() / (ref.float().norm() + 1e-30)), "cosine": float(torch.nn.functional.cosine_similarity(ref.flatten()[None].float(), got.flatten()[None].float()).item()), "nan": int(torch.isnan(got).sum()), "inf": int(torch.isinf(got).sum())}
    return out

def main():
    os.makedirs(OUT, exist_ok=True); torch.set_grad_enabled(False); dev = torch.device("cuda")
    load(CORE, "gsplat_cuda"); ext = load(TEST, "experimental_gaussian_render_inference_scene_cuda")
    result = {"so_sha256": hashlib.sha256(open(TEST, "rb").read()).hexdigest(), "scenes": {}, "timing_run": False}
    for scene in ("room", "bicycle", "garden"):
        ids, radii, means2d, depths, conics, opacity, colors, width, height = setup(scene, ext, dev); n = ids.numel(); tw, th = math.ceil(width / TS), math.ceil(height / TS)
        mm, rr, dd, cc, oo, col = means2d.reshape(1, 1, n, 2), radii.reshape(1, 1, n, 2), depths.reshape(1, 1, n), conics.reshape(1, 1, n, 3), opacity.reshape(1, 1, n), colors.reshape(1, 1, n, 3)
        _, isect, flat = torch.ops.gsplat.intersect_tile(mm, rr, dd, cc, oo, None, None, 1, TS, tw, th, True, False)
        offsets = torch.ops.gsplat.intersect_offset(isect.clone(), 1, tw, th).reshape(1, th, tw).contiguous()
        base = torch.ops.gsplat.rasterize_to_pixels_3dgs(mm, cc, col, oo, torch.zeros((1, 1, 3), device=dev), None, width, height, TS, offsets, flat.contiguous(), False, False)
        rgb, alpha, diag = torch.ops.experimental.higs_native_hierarchy_from_projected(ids, radii, means2d, depths, conics, opacity, colors, width, height, TS, torch.zeros(3, device=dev), True)
        valid = ((radii[:, 0] > 0) & (radii[:, 1] > 0)).nonzero().flatten()[:20000]; c = conics[valid]; dx = torch.tensor([-.5, .375, 1.75], device=dev); dy = torch.tensor([.25, -.625, 2.125], device=dev)
        l0 = torch.sqrt(c[:, 0]); l1 = c[:, 1] / l0; l2 = torch.sqrt(c[:, 2] - l1 * l1); qb = c[:, 0, None] * dx * dx + 2 * c[:, 1, None] * dx * dy + c[:, 2, None] * dy * dy; qt = (l0[:, None] * dx + l1[:, None] * dy) ** 2 + (l2[:, None] * dy) ** 2
        alpha_base, alpha_test = base[1], alpha
        result["scenes"][scene] = {"resolution": [width, height], "quadratic": {"n_valid": int(valid.numel()), "n_non_spd_valid": int(((c[:, 2] - l1*l1) < 0).sum()), "max_abs": float((qb-qt).abs().max()), "mean_abs": float((qb-qt).abs().mean()), "rel_l2": float((qb-qt).norm() / (qb.norm()+1e-30)), "outside_tolerance_count": int(((qb-qt).abs() > (1e-5 + 3e-5 * qb.abs())).sum())}, "rgb": stats(base[0], rgb), "alpha": stats(alpha_base, alpha_test), "support_mismatch": int(((alpha_base == 0) != (alpha_test.unsqueeze(0) == 0)).sum()), "ordering": {"non_tie_inversions": 0, "source": "R2/R3 unchanged segmented depth sort"}, "native_macro_entries": int(diag[0][-1]), "base_fine_entries": int(flat.numel())}
        print(scene, result["scenes"][scene]["rgb"]["rel_l2"], result["scenes"][scene]["alpha"]["max_abs"], flush=True)
    result["verdict"] = "P2_1A_NATIVE_HIERARCHY_DROP"
    json.dump(result, open(f"{OUT}/forward_correctness.json", "w"), indent=2)

if __name__ == "__main__": main()
