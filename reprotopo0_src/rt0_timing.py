#!/usr/bin/env python3
"""
ReproTopo0 timing driver (packet Sec.6).

Measures, per captured event:
  - per-iteration training wall (real render+backward+step, densify skipped)
    -> full_training_wall = iterations*per_iter + num_topology_events*orig_event
  - original topology event wall (densify_and_prune), 256 repeats (copy_ reset)
  - ReproTopo event wall, 256 repeats
  - peak VRAM for each
Reports:
  - max_event_wall_ratio        = reprotopo_event / original_event
  - amortized_full_training_overhead_fraction
        = num_events*(rt_event-orig_event) / full_training_wall
  - extra_vram_fraction         = (rt_peak-orig_peak)/gpu_total_bytes
  - nan_inf_count, id_collision_count (representative run)
"""
import sys
import gc
import time
import json
import argparse
import statistics
import torch

sys.path.insert(0, "/root/reprotopo0_src")
import rt0_common as C
import rt0_event as E
import rt0_statemachine as SM


# Move a CPU snapshot to a device-resident snapshot once (setup, outside
# the event timer). reprotopo_event's per-tensor .to(device) becomes a no-op on
# device tensors, so the timed region is the pure state transition -- symmetric
# with time_orig_event, which excludes restore_original_model (input loading)
# from its timer.
def preload_snapshot(snap, device):
    o = dict(snap)
    for k in ("xyz", "shs", "scaling", "rotation", "opacity", "max_radii2D",
              "xyz_gradient_accum", "denom", "gaussian_ids"):
        if torch.is_tensor(o[k]):
            o[k] = o[k].to(device)
    o["optimizer"] = {g: {m: (d[m].to(device) if d.get(m) is not None else None)
                          for m in ("exp_avg", "exp_avg_sq")}
                      for g, d in snap["optimizer"].items()}
    return o


def count_topology_events(config):
    c = 0
    for it in range(1, int(config.iterations) + 1):
        if (it > config.densify_from_iter and it < config.densify_until_iter
                and it % config.densification_interval == 0):
            c += 1
    return c


def time_per_iter(model, dataset, cam_seq, ssim_fn, config, scene_extent,
                  start_iter, n_steps, device):
    import torch.nn.functional as F
    from rt0_capture import render_with_meta
    times = []
    for s in range(n_steps + 2):
        iteration = start_iter + s
        model.update_learning_rate(iteration)
        is_densify = (iteration > config.densify_from_iter and
                      iteration < config.densify_until_iter and
                      iteration % config.densification_interval == 0)
        if is_densify:
            continue
        cam_idx = int(cam_seq[(iteration - 1) % len(cam_seq)])
        cam, gt = dataset.get_item(cam_idx)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        image, radii_full, means2d = render_with_meta(model, cam, model.active_sh_degree)
        L1 = F.l1_loss(image, gt)
        dssim = ssim_fn(image, gt)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
        loss.backward()
        radii = radii_full[0]
        vf = (radii > 0).any(dim=-1)
        if iteration < config.densify_until_iter:
            model.max_radii2D[vf] = torch.max(
                model.max_radii2D[vf], radii[vf].float().max(dim=-1).values)
            model.add_densification_stats(means2d, vf, width=cam.image_width,
                                          height=cam.image_height)
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        if s >= 2:
            times.append(t1 - t0)
    return statistics.median(times), (sum(times) / len(times) if times else 0.0)


def time_orig_event(snap, config, scene_extent, device, repeats):
    cfg = snap["config"]
    radii = snap["max_radii2D"].to(device)

    def do_event(m):
        m.densify_and_prune(
            max_grad=cfg["densify_grad_threshold"], min_opacity=cfg["min_opacity"],
            extent=scene_extent, max_screen_size=cfg["max_screen_size"], radii=radii)

    # warmup (restore + one event)
    m = E.restore_original_model(snap, device=device)
    do_event(m); del m; torch.cuda.empty_cache(); gc.collect()

    times = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        m = E.restore_original_model(snap, device=device)
        torch.cuda.synchronize(device); t0 = time.perf_counter()
        do_event(m)
        torch.cuda.synchronize(device); t1 = time.perf_counter()
        times.append(t1 - t0)
        del m
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return statistics.median(times), peak


def time_rt_event(snap, device, repeats):
    # snap is device-resident (preloaded outside the timer): .to(device) inside
    # reprotopo_event is a no-op, so only the state transition is timed.
    SM.reprotopo_event(snap, device=device); torch.cuda.synchronize(device)  # warmup
    times = []
    torch.cuda.reset_peak_memory_stats(device)
    nan_inf = 0; coll = 0
    for i in range(repeats):
        torch.cuda.synchronize(device); t0 = time.perf_counter()
        state, meta, actions = SM.reprotopo_event(snap, device=device)
        torch.cuda.synchronize(device); t1 = time.perf_counter()
        times.append(t1 - t0)
        if i == 0:
            nan_inf = int(meta["n_nan_inf"]); coll = int(meta["collision_count"])
        del state, meta, actions
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return statistics.median(times), peak, nan_inf, coll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_base", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=256)
    ap.add_argument("--n_per_iter", type=int, default=12)
    args = ap.parse_args()
    device = "cuda:%d" % args.gpu_base

    import rt0_capture as CAP
    CAP.bootstrap_c0()
    config = C.load_config()
    snap = torch.load(args.snap, map_location="cpu", weights_only=False)
    scene = snap["scene"]; scene_extent = float(snap["scene_extent"])

    # dataset + cameras (for per-iter render timing)
    ck_path = C.CELLS["%s@10000" % scene]["ckpt"]
    model10k, dataset, _se, _ck = CAP.restore_model(ck_path, config, scene, device)
    n_cameras = len(dataset)
    cam_seq = CAP.build_camera_sequence(n_cameras, config.seed, config.iterations)
    del model10k
    torch.cuda.empty_cache()

    ssim_fn = CAP.SepSSIM(device=device)

    # per-iteration training wall (real render steps on the snapshot-state model)
    m_iter = E.restore_original_model(snap, device=device)
    per_iter_median, per_iter_mean = time_per_iter(
        m_iter, dataset, cam_seq, ssim_fn, config, scene_extent,
        int(snap["iteration"]), args.n_per_iter, device)
    del m_iter
    torch.cuda.empty_cache(); gc.collect()

    # original event wall (copy_ reset + densify_and_prune)
    orig_wall, orig_peak = time_orig_event(snap, config, scene_extent, device, args.repeats)
    torch.cuda.empty_cache(); gc.collect()

    # reprotopo event wall (device-resident input; transfer excluded,
    # symmetric with time_orig_event's restore-outside-timer convention)
    snap_dev = preload_snapshot(snap, device)
    rt_wall, rt_peak, nan_inf, coll = time_rt_event(snap_dev, device, args.repeats)

    num_events = count_topology_events(config)
    iters = int(config.iterations)
    gpu_total = torch.cuda.get_device_properties(device.index).total_memory
    full_training_wall = iters * per_iter_mean + num_events * orig_wall
    extra_per_event = rt_wall - orig_wall
    amortized = (num_events * extra_per_event) / full_training_wall if full_training_wall > 0 else 0.0
    max_event_wall_ratio = rt_wall / orig_wall if orig_wall > 0 else 0.0
    extra_vram_fraction = (rt_peak - orig_peak) / gpu_total if gpu_total > 0 else 0.0

    result = {
        "scene": scene, "iteration": int(snap["iteration"]),
        "N_in": int(snap["N"]),
        "repeats": args.repeats, "n_per_iter": args.n_per_iter,
        "per_iter_wall_median_s": per_iter_median,
        "per_iter_wall_mean_s": per_iter_mean,
        "orig_event_wall_s": orig_wall,
        "reprotopo_event_wall_s": rt_wall,
        "orig_peak_bytes": int(orig_peak), "reprotopo_peak_bytes": int(rt_peak),
        "gpu_total_bytes": int(gpu_total),
        "num_topology_events": num_events, "iterations": iters,
        "full_training_wall_s": full_training_wall,
        "extra_per_event_s": extra_per_event,
        "max_event_wall_ratio": max_event_wall_ratio,
        "amortized_full_training_overhead_fraction": amortized,
        "extra_vram_fraction": extra_vram_fraction,
        "nan_inf_count": nan_inf, "id_collision_count": coll,
        "gpu_base": args.gpu_base,
    }
    C.emit_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
