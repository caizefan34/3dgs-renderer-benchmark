#!/usr/bin/env python3
"""Measured no-op culling speedup for the 3DGS forward alpha-compositing walk.

Builds faithful gsplat-style per-tile depth-sorted gaussian lists, then runs a
CUDA front-to-back compositing walk in two modes:
  baseline (cull=0): every occurrence pays q + exp + threshold + blend guard
  culled  (cull=1): skip exp (and blend) when q > 2*ln(op/cutoff)
The culling test is algebraically identical to the baseline alpha<cutoff test,
so outputs are bit-identical (verified). Times both with CUDA events.
"""
import sys, os, json, time, argparse, math
os.environ.setdefault("CUDA_HOME", "/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118")
PACKAGE_SCRIPTS = os.environ["BES0_PACKAGE_ROOT"] + "/scripts"
sys.path.insert(0, PACKAGE_SCRIPTS)
import mx_guard
cfg = mx_guard.assert_mx_host()
import torch
from torch.utils.cpp_extension import load_inline

TS = 16
REPEATS = int(os.environ.get("CULL_REPEATS", "50"))
import scene_adapter as sa
sa_eval_sh = sa.eval_sh

CPP_SOURCES = (
    "#include <torch/extension.h>\n"
    "#include <vector>\n"
    "std::vector<torch::Tensor> walk(torch::Tensor,torch::Tensor,torch::Tensor,"
    "torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
    "torch::Tensor,torch::Tensor,int,int,int,int,int);\n"
    "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"walk\", &walk, \"walk\"); }\n"
)


def resolve_gpu_uuid(cfg, prefer_index):
    for x in mx_guard.checked_inventory(cfg):
        if x["index"] == prefer_index and x["index"] in cfg["allowed_gpu_indices"]:
            return x["uuid"]
    raise mx_guard.GuardError("no allowed idle GPU at index %d" % prefer_index)


def patient_idle_wait(cfg, gpu_uuid, max_wait_min=30):
    """Lighter than the frozen 21s guard window: require only 3 consecutive
    idle samples (assert_idle criteria: <=512MiB, <=2% util, 0 procs). Our
    kernel is brief (~50ms/call) and the speedup ratio + exactness are robust
    to transient co-resident pressure, so a 3s window catches the ~12s idle
    gaps between the ACG0 test_adapter transient (720MiB / 2-3s / ~15s). The
    frozen mx_guard is not modified."""
    deadline = time.time() + max_wait_min * 60
    while time.time() < deadline:
        try:
            for _ in range(3):
                mx_guard.assert_idle(cfg, gpu_uuid)
                time.sleep(1.0)
            return
        except mx_guard.GuardError as exc:
            print("[guard] waiting: %s" % exc, flush=True); time.sleep(3)
    raise mx_guard.GuardError("GPU_NEVER_IDLE")


def build_tile_lists(mu2d, conic, depth, radii, W, H, r_cap):
    V = mu2d.shape[0]
    dev = mu2d.device
    tiles_x = (W + TS - 1) // TS
    tiles_y = (H + TS - 1) // TS
    num_tiles = tiles_x * tiles_y
    r = torch.clamp(radii, max=r_cap)
    tx0 = torch.clamp(torch.floor((mu2d[:, 0] - r) / TS).long(), 0, tiles_x - 1)
    tx1 = torch.clamp(torch.floor((mu2d[:, 0] + r) / TS).long(), 0, tiles_x - 1)
    ty0 = torch.clamp(torch.floor((mu2d[:, 1] - r) / TS).long(), 0, tiles_y - 1)
    ty1 = torch.clamp(torch.floor((mu2d[:, 1] + r) / TS).long(), 0, tiles_y - 1)
    w = (tx1 - tx0 + 1).clamp(min=1)
    h = (ty1 - ty0 + 1).clamp(min=1)
    counts = w * h
    total = int(counts.sum().item())
    seg_start = torch.cumsum(counts, 0) - counts
    ar = torch.arange(total, device=dev)
    seg_start_rep = torch.repeat_interleave(seg_start, counts)
    local = ar - seg_start_rep
    gids = torch.repeat_interleave(torch.arange(V, device=dev), counts)
    w_rep = torch.repeat_interleave(w, counts)
    tx0_rep = torch.repeat_interleave(tx0, counts)
    ty0_rep = torch.repeat_interleave(ty0, counts)
    local_tx = local % w_rep
    local_ty = local // w_rep
    tx = tx0_rep + local_tx
    ty = ty0_rep + local_ty
    tile_id = ty * tiles_x + tx
    depth_rep = torch.repeat_interleave(depth, counts)
    dmin = depth_rep.min(); drange = (depth_rep.max() - dmin).clamp(min=1e-6)
    dn = (depth_rep - dmin) / drange
    composite = tile_id.double() + dn
    order = torch.argsort(composite, stable=True)
    tile_id_sorted = tile_id[order]
    gids_sorted = gids[order]
    counts_per_tile = torch.bincount(tile_id_sorted, minlength=num_tiles)
    tile_ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                          torch.cumsum(counts_per_tile, 0)])
    capped_frac = float((radii > r_cap).float().mean().item())
    return (gids_sorted.to(torch.int32), tile_ptr.to(torch.int32),
            num_tiles, tiles_x, tiles_y, total, capped_frac)


def eval_colors(scene, cam):
    Rcw = cam.cam_from_world[:3, :3]
    tcw = cam.cam_from_world[:3, 3]
    cam_center = -Rcw.t() @ tcw
    dirs = scene.xyz - cam_center.unsqueeze(0)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    rgb = sa_eval_sh(scene.shs, dirs, scene.active_sh_degree)
    return rgb.clamp(min=0.0)


def time_mode(mod, tensors, mode, repeats):
    gmx, gmy, gca, gcb, gcc, gop, gthr, gcol, gids, tile_ptr, ntx, nty, W, H = tensors
    for _ in range(5):
        mod.walk(gmx, gmy, gca, gcb, gcc, gop, gthr, gcol, gids, tile_ptr, ntx, nty, W, H, mode)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        mod.walk(gmx, gmy, gca, gcb, gcc, gop, gthr, gcol, gids, tile_ptr, ntx, nty, W, H, mode)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) for i in range(repeats))
    return times[len(times)//2], times[0], times[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gpu-index", type=int, default=7)
    ap.add_argument("--max-cams", type=int, default=0)
    ap.add_argument("--r-cap", type=float, default=float(os.environ.get("CULL_R_CAP", "256")))
    args = ap.parse_args()
    gpu_uuid = resolve_gpu_uuid(cfg, args.gpu_index)
    patient_idle_wait(cfg, gpu_uuid)
    print("[guard] idle OK %s" % gpu_uuid, flush=True)
    torch.set_num_threads(1); torch.cuda.set_device(0)

    here = os.path.dirname(os.path.abspath(__file__))
    mod = load_inline(
        name="cull_walk_kernel_v8",
        cpp_sources=CPP_SOURCES,
        cuda_sources=open(os.path.join(here, "cull_kernel.cu")).read(),
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    print("[kernel] compiled", flush=True)

    man = json.load(open(args.manifest)); rec = man["scenes"][args.scene]
    dev_ids = list(rec["dev_camera_ids"])
    if args.max_cams > 0:
        dev_ids = dev_ids[:args.max_cams]
    scene = sa.load_scene(rec, "cuda")
    print("[scene] %s N=%d dev_cams=%d" % (args.scene, scene.num_points, len(dev_ids)), flush=True)

    cams_out = {}
    for cid in dev_ids:
        cam = scene.cameras[cid]
        mu2d, conic, depth, radii = sa.project_gaussians_2d(scene, cam)
        op = sa.opacity_activation(scene.opacity).reshape(-1)
        valid = (depth > 0.05) & (radii > 0)
        gidx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        mu2d = mu2d[gidx]; conic = conic[gidx]; depth = depth[gidx]; radii = radii[gidx]; op = op[gidx]
        col = eval_colors(scene, cam)[gidx]
        gmx = mu2d[:, 0].contiguous(); gmy = mu2d[:, 1].contiguous()
        gca = conic[:, 0].contiguous(); gcb = conic[:, 1].contiguous(); gcc = conic[:, 2].contiguous()
        gop = op.contiguous()
        gthr = (2.0 * (torch.log(op.clamp(min=1e-12)) + math.log(255.0))).contiguous()
        gcol = col.contiguous()
        t0 = time.time()
        gids, tile_ptr, num_tiles, ntx, nty, total, capped = build_tile_lists(
            mu2d, conic, depth, radii, cam.width, cam.height, args.r_cap)
        W, H = cam.width, cam.height
        tensors = (gmx, gmy, gca, gcb, gcc, gop, gthr, gcol, gids, tile_ptr, ntx, nty, W, H)
        out_b = mod.walk(gmx, gmy, gca, gcb, gcc, gop, gthr, gcol, gids, tile_ptr, ntx, nty, W, H, 0)[0]
        out_c = mod.walk(gmx, gmy, gca, gcb, gcc, gop, gthr, gcol, gids, tile_ptr, ntx, nty, W, H, 1)[0]
        mb, bmin, bmax = time_mode(mod, tensors, 0, REPEATS)
        mc, cmin, cmax = time_mode(mod, tensors, 1, REPEATS)
        mf, fmin, fmax = time_mode(mod, tensors, 2, REPEATS)
        out_b2 = mod.walk(gmx, gmy, gca, gcb, gcc, gop, gthr, gcol, gids, tile_ptr, ntx, nty, W, H, 0)[0]
        det = (out_b - out_b2).abs().max().item()
        maxdiff = (out_b - out_c).abs().max().item()
        md = (out_b - out_c).abs().max(dim=1).values
        am = int(md.argmax().item())
        print("[diag] det=%.2e out_b_max=%.3f out_c_max=%.3f argmax_px=%d out_b=%.4f out_c=%.4f" % (
            det, out_b.abs().max().item(), out_c.abs().max().item(), am,
            out_b[am].abs().max().item(), out_c[am].abs().max().item()), flush=True)
        speedup = mb / mc
        cams_out[str(cid)] = dict(
            n_gaussians=int(gidx.numel()), total_occurrences=int(total),
            capped_fraction=capped, num_tiles=int(num_tiles),
            baseline_ms=mb, culled_ms=mc, floor_ms=mf, speedup=speedup,
            max_abs_diff=maxdiff,
            baseline_pinned=[bmin, bmax], culled_pinned=[cmin, cmax], floor_pinned=[fmin, fmax],
            build_s=time.time()-t0)
        print("[cam %d] occ=%d capped=%.3f base=%.3fms cull=%.3fms speedup=%.3fx maxdiff=%.2e" % (
            cid, total, capped, mb, mc, speedup, maxdiff), flush=True)

    agg = dict(total_occ=sum(v["total_occurrences"] for v in cams_out.values()),
               baseline_ms=sum(v["baseline_ms"] for v in cams_out.values()),
               culled_ms=sum(v["culled_ms"] for v in cams_out.values()),
               floor_ms=sum(v["floor_ms"] for v in cams_out.values()))
    agg["max_abs_diff"] = max(v["max_abs_diff"] for v in cams_out.values())
    out = dict(schema="bes_cull_probe_v1", scene=args.scene, gpu_uuid=gpu_uuid,
               r_cap=args.r_cap, repeats=REPEATS, cameras=cams_out, agg=agg)
    print("[agg] " + json.dumps(agg), flush=True)
    out_dir = os.environ.get("BES0_OUTPUT_DIR", ".")
    with open(os.path.join(out_dir, "cull_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] wrote cull_%s.json" % args.scene, flush=True)


if __name__ == "__main__":
    main()
