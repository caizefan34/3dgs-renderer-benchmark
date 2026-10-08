"""Production-compositor gate counterexamples and isolated CUDA pair VJP audit.

The full-scene mode keeps the production projection, SH, parent tile coverage
and sort list, then substitutes only the compositor. No training is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(rows, width, height, offsets, ids, bg):
    from gsplat.cuda._wrapper import rasterize_to_pixels
    rgb, alpha = rasterize_to_pixels(
        rows[None, :, :2], rows[None, :, 2:5], rows[None, :, 5:8], rows[None, :, 8],
        width, height, 16, offsets.reshape(1, (height + 15) // 16, (width + 15) // 16),
        ids, backgrounds=bg[None], packed=False, absgrad=False)
    return rgb[0], alpha[0]


def delta(a, b):
    diff = (a - b).detach().double()
    ref = b.detach().double()
    return {
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "relative_l2": float(diff.norm() / ref.norm().clamp_min(1e-30)),
        "reference_norm": float(ref.norm()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def reference(base, children, mapping, width, height, bg):
    # Independent autograd oracle: explicit sequential alpha compositing within
    # each group, while group inclusion/termination decisions remain atomic.
    yy, xx = torch.meshgrid(torch.arange(height, dtype=base.dtype),
                            torch.arange(width, dtype=base.dtype), indexing="ij")
    xx, yy = xx + .5, yy + .5
    trans = torch.ones_like(xx)
    color = torch.zeros(height, width, 3, dtype=base.dtype)
    active = torch.ones_like(xx, dtype=torch.bool)
    for i in range(len(base)):
        rows = [base[i]] if mapping[i] < 0 else list(children[mapping[i]])
        alphas = []
        for row in rows:
            dx, dy = row[0] - xx, row[1] - yy
            sigma = .5 * (row[2] * dx.square() + row[4] * dy.square()) + row[3] * dx * dy
            a = (row[8] * torch.exp(-sigma)).clamp(max=.99)
            a = torch.where(sigma >= 0, a, 0.)
            alphas.append(a if len(rows) == 1 else 1 - torch.sqrt(1 - a))
        group_a = alphas[0] if len(rows) == 1 else alphas[0] + (1 - alphas[0]) * alphas[1]
        valid = group_a >= 1 / 255
        next_t = trans * (1 - group_a)
        active = active & ~(valid & (next_t <= 1e-4))
        contributes = active & valid
        local_t = trans
        local_c = color
        for a, row in zip(alphas, rows):
            local_c = local_c + (local_t * a)[..., None] * row[5:8]
            local_t = local_t * (1 - a)
        color = torch.where(contributes[..., None], local_c, color)
        trans = torch.where(contributes, local_t, trans)
    return color + trans[..., None] * bg, 1 - trans[..., None]


def synthetic_vjp(pair):
    rng = torch.Generator().manual_seed(270927)
    base = torch.tensor([
        [1.6, 1.1, .3, .04, .5, .2, .5, .8, .2],
        [2.8, 1.8, .6, -.08, .4, .8, .2, .4, .85],
        [2.0, .7, .2, .01, .3, .3, .9, .1, .4],
        [1.2, 2.0, .4, .05, .5, .7, .4, .2, .7],
        [3.4, 1.4, .7, -.03, .2, .1, .3, .6, .3],
    ], dtype=torch.float64, requires_grad=True)
    children = base.detach()[[1, 3]][:, None].repeat(1, 2, 1)
    children[:, 0, 0] += .13
    children[:, 1, 1] -= .18
    children[:, 0, 5:8] += .06
    children[:, 1, 8] -= .07
    children.requires_grad_()
    mapping = [-1, 0, -1, 1, -1]
    bg = torch.tensor([.11, .23, .07], dtype=torch.float64)
    vc = torch.rand(3, 5, 3, generator=rng, dtype=torch.float64) - .5
    va = torch.rand(3, 5, 1, generator=rng, dtype=torch.float64) - .5
    rgb, alpha = reference(base, children, mapping, 5, 3, bg)
    grads = torch.autograd.grad((rgb * vc).sum() + (alpha * va).sum(), (base, children))
    b = base.detach().float().cuda().requires_grad_()
    c = children.detach().float().cuda().requires_grad_()
    out, aout = pair.rasterize(
        b, c, torch.tensor(mapping, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        torch.arange(len(base), dtype=torch.int32, device="cuda"),
        5, 3, bg.float().cuda())
    actual = torch.autograd.grad((out * vc.float().cuda()).sum() +
                                 (aout * va.float().cuda()).sum(), (b, c))
    result = {"rgb": delta(out.cpu(), rgb), "alpha": delta(aout.cpu(), alpha),
              "base_vjp": delta(actual[0].cpu(), grads[0]),
              "independent_children_vjp": delta(actual[1].cpu(), grads[1])}
    result["passed"] = all(v["finite"] and v["max_abs"] < 3e-6 and v["relative_l2"] < 1e-4
                           for v in result.values())
    return result


def gate_cases(pair):
    records = []
    for name, a, prefixes in [
        ("ordinary", .6, [.1, .2]),
        ("child_alpha_cutoff", .006, [0., 0.]),
        ("parent_alpha_clamp", .999, [0., 0.]),
        ("between_children_early_exit", .95, [.9, .99]),
        ("saturated_opacity", 1., [0., 0.]),
    ]:
        bg = torch.tensor([.13, .07, .19], device="cuda")
        base = torch.tensor([
            [.5, .5, 1., 0., 1., .2, .3, .4, prefixes[0]],
            [.5, .5, 1., 0., 1., .4, .1, .2, prefixes[1]],
            [.5, .5, 1., 0., 1., .9, .2, .1, a],
            [.5, .5, 1., 0., 1., .1, .8, .3, .2],
        ], device="cuda", requires_grad=True)
        offsets = torch.zeros(1, dtype=torch.int64, device="cuda")
        out = {}
        for mode in ("parent", "independent_raw_h", "independent_clamped_h", "atomic_pair"):
            if mode == "parent":
                rgb, alpha = native(base, 1, 1, offsets,
                                    torch.arange(4, dtype=torch.int32, device="cuda"), bg)
            elif mode == "atomic_pair":
                rgb, alpha = pair.rasterize(
                    base, base[2:3, None].expand(1, 2, 9).contiguous(),
                    torch.tensor([-1, -1, 0, -1], dtype=torch.int32, device="cuda"),
                    offsets, torch.arange(4, dtype=torch.int32, device="cuda"), 1, 1, bg)
            else:
                value = base[2, 8]
                if mode == "independent_clamped_h":
                    value = value.clamp(max=.99)
                h = value / (1 + torch.sqrt(1 - value))
                row = torch.cat((base[2, :8], h[None]))
                rows = torch.cat((base[:2], row[None].expand(2, 9), base[3:]))
                rgb, alpha = native(rows, 1, 1, offsets,
                                    torch.arange(5, dtype=torch.int32, device="cuda"), bg)
            grad = torch.autograd.grad(rgb.sum() + .37 * alpha.sum(), base, retain_graph=True)[0]
            out[mode] = {"rgb": rgb.detach().flatten().tolist(), "alpha": float(alpha),
                         "parent_opacity_gradient": float(grad[2, 8]) if torch.isfinite(grad[2, 8]) else None,
                         "finite_vjp": bool(torch.isfinite(grad).all())}
        for mode in list(out)[1:]:
            out[mode]["max_rgb_difference_from_parent"] = max(
                abs(x - y) for x, y in zip(out[mode]["rgb"], out["parent"]["rgb"]))
        records.append({"case": name, "raw_parent_alpha": a, "modes": out})
    return records


def gradient_metrics(grads, reference_grads, labels):
    result = {name: delta(a, b) for name, a, b in zip(labels, grads, reference_grads)}
    for name, sl in (("screen_xy", slice(0, 2)), ("screen_conic", slice(2, 5)),
                     ("rgb", slice(5, 8)), ("opacity", slice(8, 9))):
        result[name] = delta(grads[0][:, sl], reference_grads[0][:, sl])
    return result


def real_scene(args, common, pair, assert_exclusive):
    from gaussian_model import GaussianModel
    from sort_reuse_renderer import prepare, intersections
    root = Path(__file__).parent
    state_path = Path(common.STATEGEN_DIR) / args.scene / "state_2000.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model = GaussianModel(max_sh_degree=3)
    model.restore(state, common.TRAIN_CFG)
    del state
    dataset = common.load_dataset(args.scene)
    seq = np.load(state_path.with_name("camera_sequence.npy"))
    views = list(dict.fromkeys(seq[2000:].tolist()))[:args.views]
    selected = torch.tensor(json.loads((root / "room_split_candidate_ids.json").read_text()),
                            dtype=torch.long, device="cuda")
    assert args.scene == "room", "candidate ids are from room@2000 only"
    result = {"checkpoint_sha256": sha(state_path), "scene": args.scene, "views": [],
              "scope": "fixed production projection/SH/tile coverage/depth order; no training",
              "gate": {"image_max_abs": 2e-5, "image_mean_abs": 1e-6,
                       "vjp_relative_l2": 1e-3, "or_native_replay_envelope_factor": 3,
                       "tiny_reference_norm": 1e-10, "tiny_vjp_max_abs": 1e-8}}
    labels = ["all_projected", "xyz", "log_scale", "rotation", "opacity_logit", "sh_dc", "sh_rest"]
    for camera_id in views:
        cam, gt = dataset.get_item(camera_id)
        data = prepare(model, cam)
        lists = intersections(data, cam.image_width, cam.image_height)
        base = torch.cat((data["xy"][0], data["conics"][0], data["colors"][0],
                          data["opacity"].reshape(-1, 1)), dim=-1).contiguous()
        n = len(base)
        targets = [base, model._xyz, model._scaling, model._rotation,
                   model._opacity, model._features_dc, model._features_rest]
        bg = torch.zeros(3, device="cuda")
        offsets, ids = lists[3], lists[2]

        def evaluate(mode):
            if mode == "native":
                image, alpha = native(base, cam.image_width, cam.image_height, offsets, ids, bg)
            else:
                chosen = selected if mode == "pair256" else torch.arange(n, device="cuda")
                if mode == "single":
                    chosen = chosen[:0]
                children = base[chosen, None].expand(-1, 2, -1).contiguous()
                mapping = torch.full((n,), -1, dtype=torch.int32, device="cuda")
                mapping[chosen] = torch.arange(len(chosen), dtype=torch.int32, device="cuda")
                image, alpha = pair.rasterize(base, children, mapping, offsets, ids,
                                              cam.image_width, cam.image_height, bg)
            loss = common.fused_total_loss(image.clamp(0, 1), gt)
            gradients = torch.autograd.grad(loss, targets, retain_graph=True)
            return image.detach(), alpha.detach(), float(loss), [g.detach() for g in gradients]

        reference_image, reference_alpha, reference_loss, reference_grads = evaluate("native")
        replay_image, replay_alpha, replay_loss, replay_grads = evaluate("native")
        replay = gradient_metrics(replay_grads, reference_grads, labels)
        view = {"camera_id": camera_id, "width": cam.image_width, "height": cam.image_height,
                "N": n, "tile_intersections": ids.numel(), "native_loss": reference_loss,
                "native_replay": {"image": delta(replay_image, reference_image),
                                   "alpha": delta(replay_alpha, reference_alpha), "vjp": replay}, "arms": {}}
        for mode in ("single", "pair256", "pair_all"):
            assert_exclusive()
            image, alpha, loss, grads = evaluate(mode)
            metrics = gradient_metrics(grads, reference_grads, labels)
            diff_image = delta(image, reference_image)
            checks = {}
            for name, m in metrics.items():
                checks[name] = m["finite"] and (
                    m["max_abs"] <= 1e-8 if m["reference_norm"] < 1e-10 else
                    m["relative_l2"] <= max(1e-3, 3 * replay[name]["relative_l2"]))
            samples = []
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                measured = evaluate(mode)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
                del measured
            view["arms"][mode] = {
                "image": diff_image, "alpha": delta(alpha, reference_alpha),
                "loss": loss, "loss_delta": loss - reference_loss, "vjp": metrics,
                "vjp_gate_by_parameter": checks,
                "image_gate": diff_image["finite"] and diff_image["max_abs"] <= 2e-5 and diff_image["mean_abs"] <= 1e-6,
                "diagnostic_raster_loss_vjp_ms": samples,
                "timing_scope": "naive diagnostic kernel plus child construction, fused loss and VJP; not optimized training speed",
            }
            print(f"view={camera_id} mode={mode} image_max={diff_image['max_abs']:.3g} "
                  f"vjp_gate={all(checks.values())} median_ms={np.median(samples):.1f}", flush=True)
        result["views"].append(view)
        del data, lists, base, targets, reference_grads, replay_grads, grads
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--scene", default="room")
    ap.add_argument("--views", type=int, default=1)
    ap.add_argument("--full-scene", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    os.environ.setdefault("MAX_JOBS", "1")
    torch.set_num_threads(4)
    sys.path.insert(0, "/root/epic05_p11")
    import p11_common as common
    gpu = common.assert_clean_gpu(args.gpu)
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    uuid = gpu["gpu_info"].split(",")[0].strip()
    owners = {row.split(",")[0].strip() for row in common.gpu_compute_apps() if uuid in row}
    assert len(owners) == 1, owners

    def assert_exclusive():
        current = {row.split(",")[0].strip() for row in common.gpu_compute_apps() if uuid in row}
        assert current == owners, (current, owners)

    identity = common.bootstrap_gmain()
    common.import_recipe()
    import survival_pair_renderer as pair
    root = Path(__file__).parent
    result = {"completed": False, "gpu": gpu, "renderer_identity": identity,
              "sources": {name: sha(root / name) for name in
                          ("survival_pair_probe.py", "survival_pair_renderer.py", "survival_pair_cuda.cu",
                           "sort_reuse_renderer.py", "room_split_candidate_ids.json")},
              "native_sources": {name: sha(Path(common.GMAIN_TREE) / "gsplat/cuda" / name) for name in
                                 ("include/Common.h", "csrc/RasterizeToPixels3DGSDevice.cuh",
                                  "csrc/RasterizeToPixels3DGSSerialBatchFwd.cu")}}
    started = time.perf_counter()
    pair.extension()
    result["synthetic_independent_child_vjp"] = synthetic_vjp(pair)
    result["native_gate_cases"] = gate_cases(pair)
    output = Path(args.out)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    assert result["synthetic_independent_child_vjp"]["passed"], result["synthetic_independent_child_vjp"]
    assert all(r["modes"]["atomic_pair"]["finite_vjp"] and
               r["modes"]["atomic_pair"]["max_rgb_difference_from_parent"] < 2e-6
               for r in result["native_gate_cases"]), result["native_gate_cases"]
    print("Synthetic independent-child VJP and native gate cases completed", flush=True)
    if args.full_scene:
        result["real_scene"] = real_scene(args, common, pair, assert_exclusive)
    result["completed"] = True
    result["wall_seconds_including_build"] = time.perf_counter() - started
    result["extension_sha256"] = sha(pair.extension().__file__)
    assert_exclusive()
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"completed: {output}", flush=True)


if __name__ == "__main__":
    main()
