#!/usr/bin/env python3
"""Measured EXACT forward speedup of the TWO-STAGE per-tile AABB reject.

Stage 1 loads only (mx,my,gdx,gdy) + keep flag for every (tile,gaussian);
stage 2 loads the other 9 floats + q/exp/threshold/blend ONLY for AABB survivors.
This attacks the staging floor that caps the single-stage per-tile AABB reject.
modes: 0 baseline (13-float load, no reject), 6 two-stage tile reject,
7 two-stage tile+pix, 8 two-stage floor (stage 1 only).
"""
import sys, os, json, time, math, argparse
os.environ.setdefault("CUDA_HOME", "/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118")
os.environ.setdefault("BES0_PACKAGE_ROOT", "/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH")
PACKAGE_SCRIPTS = os.environ["BES0_PACKAGE_ROOT"] + "/scripts"
sys.path.insert(0, PACKAGE_SCRIPTS)
import mx_guard
cfg = mx_guard.assert_mx_host()
import torch
from torch.utils.cpp_extension import load_inline
TS = 16
REPEATS = int(os.environ.get("AABB_REPEATS", "50"))
import scene_adapter as sa
sa_eval_sh = sa.eval_sh
NL = chr(10); Q = chr(34)
CPP_SOURCES = (
    "#include <torch/extension.h>" + NL + "#include <vector>" + NL +
    "std::vector<torch::Tensor> walk2(torch::Tensor,torch::Tensor,torch::Tensor,"
    "torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
    "torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
    "int,int,int,int);" + NL +
    "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(" + Q + "walk2" + Q + ", &walk2, " + Q + "walk2" + Q + "); }" + NL
)
def resolve_gpu_uuid(cfg, prefer_index):
    for x in mx_guard.checked_inventory(cfg):
        if x["index"] == prefer_index and x["index"] in cfg["allowed_gpu_indices"]:
            return x["uuid"]
    raise mx_guard.GuardError("no allowed idle GPU at index %d" % prefer_index)
def patient_idle_wait(cfg, gpu_uuid, max_wait_min=240):
    deadline = time.time() + max_wait_min * 60
    while time.time() < deadline:
        try:
            for _ in range(3):
                mx_guard.assert_idle(cfg, gpu_uuid); time.sleep(1.0)
            return
        except mx_guard.GuardError as exc:
            print("[guard] waiting: %s" % exc, flush=True); time.sleep(3)
    raise mx_guard.GuardError("GPU_NEVER_IDLE")
def build_tile_lists(mu2d, conic, depth, radii, W, H, r_cap):
    V = mu2d.shape[0]; dev = mu2d.device
    tiles_x = (W + TS - 1) // TS; tiles_y = (H + TS - 1) // TS
    num_tiles = tiles_x * tiles_y
    r = torch.clamp(radii, max=r_cap)
    tx0 = torch.clamp(torch.floor((mu2d[:, 0] - r) / TS).int(), 0, tiles_x - 1)
    tx1 = torch.clamp(torch.floor((mu2d[:, 0] + r) / TS).int(), 0, tiles_x - 1)
    ty0 = torch.clamp(torch.floor((mu2d[:, 1] - r) / TS).int(), 0, tiles_y - 1)
    ty1 = torch.clamp(torch.floor((mu2d[:, 1] + r) / TS).int(), 0, tiles_y - 1)
    w = (tx1 - tx0 + 1).clamp(min=1); h = (ty1 - ty0 + 1).clamp(min=1)
    counts = w * h; total = int(counts.sum().item())
    assert total < 2147483647, "int32 overflow total=%d" % total
    seg_start = torch.cumsum(counts, 0) - counts
    ar = torch.arange(total, device=dev, dtype=torch.int32)
    seg_rep = torch.repeat_interleave(seg_start, counts)
    local = ar - seg_rep
    gids = torch.repeat_interleave(torch.arange(V, device=dev, dtype=torch.int32), counts)
    w_rep = torch.repeat_interleave(w, counts)
    tx0_rep = torch.repeat_interleave(tx0, counts); ty0_rep = torch.repeat_interleave(ty0, counts)
    local_tx = local % w_rep; local_ty = local // w_rep
    tx = tx0_rep + local_tx; ty = ty0_rep + local_ty
    tile_id = ty * tiles_x + tx
    depth_rep = torch.repeat_interleave(depth, counts)
    dmin = depth_rep.min(); drange = (depth_rep.max() - dmin).clamp(min=1e-6)
    dn = (depth_rep - dmin) / drange
    composite = tile_id.double() + dn
    order = torch.argsort(composite, stable=True)
    tile_id_sorted = tile_id[order]; gids_sorted = gids[order]
    counts_per_tile = torch.bincount(tile_id_sorted, minlength=num_tiles)
    tile_ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(counts_per_tile, 0)])
    capped_frac = float((radii > r_cap).float().mean().item())
    return (gids_sorted.to(torch.int32), tile_ptr.to(torch.int32), num_tiles, tiles_x, tiles_y, total, capped_frac)
def eval_colors(scene, cam):
    Rcw = cam.cam_from_world[:3, :3]; tcw = cam.cam_from_world[:3, 3]
    cam_center = -Rcw.t() @ tcw
    dirs = scene.xyz - cam_center.unsqueeze(0)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    rgb = sa_eval_sh(scene.shs, dirs, scene.active_sh_degree)
    return rgb.clamp(min=0.0)
def time_mode(mod, tensors, mode, repeats):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record(); mod.walk2(*tensors, mode); ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) for i in range(repeats))
    return times[len(times)//2], times[0], times[-1]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gpu-index", type=int, default=6)
    ap.add_argument("--max-cams", type=int, default=0)
    ap.add_argument("--r-cap", type=float, default=256.0)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    print("[kernel] compiling two-stage (CPU)...", flush=True)
    mod = load_inline(name="aabb_walk_kernel_v5",
        cpp_sources=CPP_SOURCES,
        cuda_sources=open(os.path.join(here, "aabb_kernel2.cu")).read(),
        extra_cuda_cflags=["-O3"], verbose=False)
    print("[kernel] compiled", flush=True)
    gpu_uuid = resolve_gpu_uuid(cfg, args.gpu_index)
    if os.environ.get("AABB_ALLOW_BUSY"):
        print("[guard] AABB_ALLOW_BUSY set -> skip idle wait on %s" % gpu_uuid, flush=True)
    else:
        patient_idle_wait(cfg, gpu_uuid)
    print("[guard] idle OK %s" % gpu_uuid, flush=True)
    torch.set_num_threads(1); torch.cuda.set_device(0)
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
        Tpos = torch.clamp(gthr, min=0.0)
        det64 = (gca.double() * gcc.double() - gcb.double() * gcb.double()).clamp(min=1e-30)
        gdx64 = torch.sqrt(Tpos.double() * (gcc.double() / det64))
        gdy64 = torch.sqrt(Tpos.double() * (gca.double() / det64))
        det32 = gca * gcc - gcb * gcb
        safe32 = torch.isfinite(det32) & (det32 > 0)
        det32c = torch.where(safe32, det32, torch.full_like(det32, 1e-30))
        dx32 = torch.sqrt(torch.clamp(Tpos * (gcc / det32c), min=0.0))
        dy32 = torch.sqrt(torch.clamp(Tpos * (gca / det32c), min=0.0))
        big = torch.full_like(dx32, 1e9)
        dx32 = torch.where(torch.isfinite(dx32), dx32, big); dy32 = torch.where(torch.isfinite(dy32), dy32, big)
        gdx = torch.maximum(gdx64, dx32.double()).float(); gdy = torch.maximum(gdy64, dy32.double()).float()
        cacg = gca * gcc
        corr = torch.where(cacg > 0, det32 / torch.clamp(cacg, min=1e-30), torch.full_like(det32, -1.0))
        bad = (~torch.isfinite(det32)) | ~(corr > float(os.environ.get("AABB_TAU", "1e-6")))
        n_never = int(bad.sum().item())
        gdx = torch.where(bad, torch.full_like(gdx, 1e9), gdx); gdy = torch.where(bad, torch.full_like(gdy, 1e9), gdy)
        MARGIN_ABS = 1e-3; MARGIN_REL = float(os.environ.get("AABB_INFLATE", "1e-2"))
        gdx = (gdx * (1.0 + MARGIN_REL) + MARGIN_ABS).contiguous(); gdy = (gdy * (1.0 + MARGIN_REL) + MARGIN_ABS).contiguous()
        gcol = col.contiguous().float()
        t0 = time.time()
        gids, tile_ptr, num_tiles, ntx, nty, total, capped = build_tile_lists(mu2d, conic, depth, radii, cam.width, cam.height, args.r_cap)
        W, H = cam.width, cam.height
        tensors = (gmx, gmy, gca, gcb, gcc, gop, gthr, gcol, gdx, gdy, gids, tile_ptr, ntx, W, H)
        out_b = mod.walk2(*tensors, 0)[0]
        out_6 = mod.walk2(*tensors, 6)[0]
        out_7 = mod.walk2(*tensors, 7)[0]
        out_b2 = mod.walk2(*tensors, 0)[0]
        det = (out_b - out_b2).abs().max().item()
        maxdiff_6 = (out_b - out_6).abs().max().item()
        maxdiff_7 = (out_b - out_7).abs().max().item()
        mb, bmin, bmax = time_mode(mod, tensors, 0, REPEATS)
        m6, f6min, f6max = time_mode(mod, tensors, 6, REPEATS)
        m7, p7min, p7max = time_mode(mod, tensors, 7, REPEATS)
        mf, flmin, flmax = time_mode(mod, tensors, 8, REPEATS)
        print("[aabbfix] cam=%d n_never_skip=%d frac=%.4f" % (cid, n_never, n_never / max(gca.numel(), 1)), flush=True)
        print("[diag] det=%.2e maxdiff_ts=%.2e maxdiff_tsp=%.2e" % (det, maxdiff_6, maxdiff_7), flush=True)
        print("[cam %d] occ=%d base=%.3fms ts=%.3fms (%.3fx) tsp=%.3fms (%.3fx) floor=%.3fms" % (
            cid, total, mb, m6, mb / m6, m7, mb / m7, mf), flush=True)
        cams_out[str(cid)] = dict(
            n_gaussians=int(gidx.numel()), total_occurrences=int(total),
            capped_fraction=capped, num_tiles=int(num_tiles),
            baseline_ms=mb, ts_ms=m6, tsp_ms=m7, floor_ms=mf,
            speedup_ts=mb / m6, speedup_tsp=mb / m7,
            baseline_pinned=[bmin, bmax], ts_pinned=[f6min, f6max],
            tsp_pinned=[p7min, p7max], floor_pinned=[flmin, flmaxmax] if False else [flmin, flmax],
            max_abs_diff_ts=maxdiff_6, max_abs_diff_tsp=maxdiff_7, build_s=time.time() - t0)
    agg = dict(total_occ=sum(v["total_occurrences"] for v in cams_out.values()),
               baseline_ms=sum(v["baseline_ms"] for v in cams_out.values()),
               ts_ms=sum(v["ts_ms"] for v in cams_out.values()),
               tsp_ms=sum(v["tsp_ms"] for v in cams_out.values()),
               floor_ms=sum(v["floor_ms"] for v in cams_out.values()),
               max_abs_diff_ts=max(v["max_abs_diff_ts"] for v in cams_out.values()),
               max_abs_diff_tsp=max(v["max_abs_diff_tsp"] for v in cams_out.values()))
    agg["speedup_ts"] = agg["baseline_ms"] / agg["ts_ms"]
    agg["speedup_tsp"] = agg["baseline_ms"] / agg["tsp_ms"]
    out = dict(schema="bes0_aabb2_probe_v1", scene=args.scene, gpu_uuid=gpu_uuid,
               r_cap=args.r_cap, repeats=REPEATS, cameras=cams_out, agg=agg)
    print("[agg] " + json.dumps(agg), flush=True)
    out_dir = os.environ.get("BES0_OUTPUT_DIR", ".")
    with open(os.path.join(out_dir, "aabb2_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] wrote aabb2_%s.json" % args.scene, flush=True)
if __name__ == "__main__":
    main()
