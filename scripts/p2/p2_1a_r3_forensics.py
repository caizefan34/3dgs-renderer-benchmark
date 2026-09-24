#!/usr/bin/env python3
"""R3 pre-repair per-splat audit against frozen C0/F9 state; correctness only."""
import argparse, importlib.util, json, math, os
import torch

CORE = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
R2 = "/mnt/storage_pool/liaoyuanjun/higs_p2_1a_r2_final_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CKPT = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7"
CAMS = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360"
MAX_LONG, TS = 2048, 16

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def camera(camera, device):
    rot = torch.tensor(camera["rotation"], dtype=torch.float32, device=device)
    pos = torch.tensor(camera["position"], dtype=torch.float32, device=device)
    scale = MAX_LONG / max(float(camera["width"]), float(camera["height"]))
    width, height = int(round(camera["width"] * scale)), int(round(camera["height"] * scale))
    view = torch.eye(4, dtype=torch.float32, device=device); view[:3, :3] = rot; view[:3, 3] = -rot @ pos
    K = torch.tensor([[camera["fx"] * scale, 0, width / 2 - .5], [0, camera["fy"] * scale, height / 2 - .5], [0, 0, 1]], dtype=torch.float32, device=device)[None]
    return view[None], K, width, height

def f9(scene, ext, device):
    cam = json.load(open(f"{CAMS}/{scene}/cameras.json"))[0]
    ck = torch.load(f"{CKPT}/a100_30k_{scene}_t16_16/a100_30k_{scene}_t16_16_latest.pt", map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck); view, K, width, height = camera(cam, device)
    means = state["xyz"].to(device).contiguous(); n = means.shape[0]
    ids = torch.arange(n, device=device, dtype=torch.int64)
    vals = ext.higs_gatherless_projected_producer(ids, means, state["rotations"].to(device).contiguous(), torch.exp(state["scales"].to(device)).contiguous(), torch.sigmoid(state["opacity"].to(device).flatten()).contiguous(), state["shs"].to(device).contiguous(), view.contiguous(), K.contiguous(), ext.higs_camera_positions_from_viewmats(view.contiguous()), width, height, .3, .01, 1e10, 0.)
    return ids, *vals, width, height

def metric(diff, ref):
    return {"max_abs": float(diff.abs().max().item()), "mean_abs": float(diff.abs().mean().item()), "rel_l2": float(diff.norm().item() / (ref.norm().item() + 1e-30)), "outside_tolerance_count": int((diff.abs() > (1e-5 + 3e-5 * ref.abs())).sum().item())}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True); parser.add_argument("--so", default=R2); parser.add_argument("--trace-count", type=int, default=50)
    args = parser.parse_args(); os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False); device = torch.device("cuda"); load(CORE, "gsplat_cuda"); ext = load(args.so, "experimental_gaussian_render_inference_scene_cuda")
    all_q, traces = {}, {}
    for scene in ("room", "bicycle", "garden"):
        ids, radii, means2d, depths, conics, opacity, colors, width, height = f9(scene, ext, device)
        valid = (radii[:, 0] > 0) & (radii[:, 1] > 0)
        rows = valid.nonzero().flatten(); sample = rows[:min(20000, rows.numel())]
        offsets = torch.tensor([[-3.25, -1.5], [-.5, .25], [.375, -.625], [1.75, 2.125]], device=device)
        c = conics[sample]; dx, dy = offsets[:, 0], offsets[:, 1]
        q_base = c[:, 0, None] * dx * dx + 2 * c[:, 1, None] * dx * dy + c[:, 2, None] * dy * dy
        l0 = torch.sqrt(torch.clamp_min(c[:, 0], 0.)); l1 = torch.where(l0 > 1e-12, c[:, 1] / l0, torch.zeros_like(l0)); l2 = torch.sqrt(torch.clamp_min(c[:, 2] - l1 * l1, 0.))
        q_r2 = (l0[:, None] * dx + l1[:, None] * dy) ** 2 + (l2[:, None] * dy) ** 2
        all_q[scene] = {"n_valid_projected": int(rows.numel()), "n_non_spd_valid": int(((c[:, 2] - l1 * l1) < 0).sum().item()), **metric(q_r2 - q_base, q_base)}
        if scene != "room": continue
        tw, th = math.ceil(width / TS), math.ceil(height / TS)
        n = ids.numel(); mm = means2d.reshape(1, 1, n, 2); rr = radii.reshape(1, 1, n, 2); dd = depths.reshape(1, 1, n); cc = conics.reshape(1, 1, n, 3); oo = opacity.reshape(1, 1, n); col = colors.reshape(1, 1, n, 3)
        _, isect, flat = torch.ops.gsplat.intersect_tile(mm, rr, dd, cc, oo, None, None, 1, TS, tw, th, True, False)
        offsets_base = torch.ops.gsplat.intersect_offset(isect.clone(), 1, tw, th).reshape(1, th, tw).contiguous()
        base_out = torch.ops.gsplat.rasterize_to_pixels_3dgs(mm, cc, col, oo, torch.zeros((1, 1, 3), device=device), None, width, height, TS, offsets_base, flat.contiguous(), False, False)
        base_rgb, base_alpha = base_out[0], base_out[1]
        native_rgb, native_alpha, diagnostic = torch.ops.experimental.higs_native_hierarchy_from_projected(ids, radii, means2d, depths, conics, opacity, colors, width, height, TS, torch.zeros(3, device=device), True)
        # documented high-signal corner tile; select its pixel with maximum base alpha.
        tile = 0; tx, ty = tile % tw, tile // tw
        crop = base_alpha[0, ty * TS:(ty + 1) * TS, tx * TS:(tx + 1) * TS, 0]
        local = int(crop.reshape(-1).argmax().item()); px, py = tx * TS + local % TS, ty * TS + local // TS
        n_tiles = tw * th; bits = int(math.floor(math.log2(n_tiles))) + 1; tid = ((isect.cpu() >> 32) & ((1 << bits) - 1)).to(torch.int64)
        gids = flat.cpu()[tid == tile].tolist()
        gids = sorted(gids, key=lambda g: float(depths[g]))
        rows_trace, tb, tt = [], 1.0, 1.0
        for gid in gids:
            if len(rows_trace) >= args.trace_count: break
            dxv, dyv = float(means2d[gid, 0] - px), float(means2d[gid, 1] - py)
            a, b, c2 = (float(v) for v in conics[gid].tolist()); op = float(opacity[gid])
            qb = a * dxv * dxv + 2 * b * dxv * dyv + c2 * dyv * dyv; sb = .5 * qb; ab0 = op * math.exp(-sb); ab = min(.999, ab0); valid_b = sb >= 0 and ab >= 1 / 255
            l0v = math.sqrt(max(a, 0.)); l1v = b / l0v if l0v > 1e-12 else 0.; l2v = math.sqrt(max(c2 - l1v * l1v, 0.)); q2 = (l0v * dxv + l1v * dyv) ** 2 + (l2v * dyv) ** 2
            st = math.log2(math.e) * .5 * q2 - math.log2(op); at0 = 2 ** (-st); valid_t = st < math.log2(255.)
            wb, wt = (tb * ab if valid_b else 0.), (tt * at0 if valid_t else 0.)
            color = [float(v) for v in colors[gid].tolist()]
            rows_trace.append({"gid": int(ids[gid]), "depth": float(depths[gid]), "dx": dxv, "dy": dyv, "conic_raw": [a, b, c2], "base": {"q": qb, "sigma": sb, "opacity_input": op, "alpha_before_clamp": ab0, "alpha_after_clamp": ab, "support_predicate": valid_b, "T_before": tb, "weight": wb, "weight_color": [wb * x for x in color], "T_after": tb * (1 - ab) if valid_b else tb, "termination": (tb * (1 - ab) <= 1e-4) if valid_b else False}, "r2": {"q": q2, "sigma_log2": st, "opacity_input": op, "alpha_before_clamp": at0, "alpha_after_clamp": at0 if valid_t else 0., "support_predicate": valid_t, "T_before": tt, "weight": wt, "weight_color": [wt * x for x in color], "T_after": tt - wt if valid_t else tt, "termination": (tt - wt < 1e-4) if valid_t else False}, "rgb_source_color": color, "first_difference": "q" if abs(qb-q2)>1e-5 else ("alpha" if abs(ab-at0)>1e-5 else ("T_before" if abs(tb-tt)>1e-5 else "none"))})
            if valid_b: tb *= 1 - ab
            if valid_t: tt -= wt
        # Exercise the actual native queue with one-hot source colors.  Opacity/order
        # stay unchanged, so the selected output channel is that gid's true weight.
        isolated = []
        for entry in rows_trace[:20]:
            gid = entry["gid"]
            one_hot = torch.zeros_like(colors); one_hot[gid, 0] = 1.
            irgb, ialpha, _ = torch.ops.experimental.higs_native_hierarchy_from_projected(ids, radii, means2d, depths, conics, opacity, one_hot, width, height, TS, torch.zeros(3, device=device), False)
            entry["r2"]["actual_native_weight_one_hot"] = float(irgb[py, px, 0])
            entry["r2"]["actual_native_alpha_one_hot"] = float(ialpha[py, px, 0])
            isolated.append(gid)
        native_pixel = native_rgb[py, px]
        tile_rgb = base_rgb[0, :TS, :TS]
        closest = int(((tile_rgb - native_pixel).square().sum(dim=-1)).reshape(-1).argmin().item())
        traces[scene] = {"tile": tile, "pixel": [px, py], "base_alpha": float(base_alpha[0, py, px, 0]), "r2_alpha": float(native_alpha[py, px, 0]), "base_rgb": [float(v) for v in base_rgb[0, py, px].tolist()], "r2_rgb": [float(v) for v in native_pixel.tolist()], "nearest_base_pixel_to_r2_rgb_in_tile": [closest % TS, closest // TS], "nearest_base_rgb": [float(v) for v in tile_rgb[closest // TS, closest % TS].tolist()], "n_gids": len(gids), "one_hot_native_gids": isolated, "entries": rows_trace}
    json.dump(all_q, open(os.path.join(args.out, "quadratic_pre_repair.json"), "w"), indent=2)
    json.dump(traces, open(os.path.join(args.out, "per_splat_trace_r2.json"), "w"), indent=2)

if __name__ == "__main__": main()
