#!/usr/bin/env python3
"""Measured exact forward speedup of the PER-TILE AABB REJECT (modes 3/4).

Builds faithful gsplat-style per-tile depth-sorted gaussian lists (same as
cull_probe), then runs a CUDA front-to-back compositing walk:
  mode 0 baseline : every occurrence pays q + exp + threshold + blend
  mode 3 tile     : skip whole (tile,gaussian) if its 1/255 AABB misses the tile
  mode 4 tile+pix : mode 3 + per-pixel |dx|>dx_max || |dy|>dy_max reject before q
The AABB reject is bit-identical to the baseline (verified: max|diff|=0, det=0).
Times all modes with CUDA events.
"""
import sys, os, json, time, argparse, math
os.environ.setdefault("CUDA_HOME", "/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118")
PACKAGE_SCRIPTS = os.environ["BES0_PACKAGE_ROOT"] + "/scripts"
sys.path.insert(0, PACKAGE_SCRIPTS)
import mx_guard
cfg = mx_guard.assert_mx_host()
import torch
from torch.utils.cpp_extension import load_inline


TS = int(os.environ.get("AABB_TILE_SIZE", "16"))
REPEATS = int(os.environ.get("AABB_REPEATS", "50"))
import scene_adapter as sa
sa_eval_sh = sa.eval_sh

CPP_SOURCES = (
    "#include <torch/extension.h>\n"
    "#include <vector>\n"
    "std::vector<torch::Tensor> walk(torch::Tensor,torch::Tensor,torch::Tensor,"
    "torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
    "torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
    "int,int,int,int,int,int);\n"
    "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"walk\", &walk, \"walk\"); }\n"
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
    # int32 index tensors halve transient memory vs int64 (total < 2^31 for these scenes).
    tx0 = torch.clamp(torch.floor((mu2d[:, 0] - r) / TS).int(), 0, tiles_x - 1)
    tx1 = torch.clamp(torch.floor((mu2d[:, 0] + r) / TS).int(), 0, tiles_x - 1)
    ty0 = torch.clamp(torch.floor((mu2d[:, 1] - r) / TS).int(), 0, tiles_y - 1)
    ty1 = torch.clamp(torch.floor((mu2d[:, 1] + r) / TS).int(), 0, tiles_y - 1)
    w = (tx1 - tx0 + 1).clamp(min=1)
    h = (ty1 - ty0 + 1).clamp(min=1)
    counts = w * h
    total = int(counts.sum().item())
    assert total < 2147483647, "int32 overflow total=%d" % total
    seg_start = torch.cumsum(counts, 0) - counts
    ar = torch.arange(total, device=dev, dtype=torch.int32)
    seg_start_rep = torch.repeat_interleave(seg_start, counts)
    local = ar - seg_start_rep
    gids = torch.repeat_interleave(torch.arange(V, device=dev, dtype=torch.int32), counts)
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
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        mod.walk(*tensors, mode, TS)
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
    ap.add_argument("--r-cap", type=float, default=float(os.environ.get("AABB_R_CAP", "256")))
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    print("[kernel] compiling (CPU)...", flush=True)
    mod = load_inline(
        name=os.environ.get("AABB_MODULE", "aabb_walk_kernel_ts"),
        cpp_sources=CPP_SOURCES,
        cuda_sources=open(os.path.join(here, os.environ.get("AABB_KERNEL_CU", "aabb_kernel_ts.cu"))).read(),
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    print("[kernel] compiled TS=%d" % TS, flush=True)
    print("[tsconfig] TS=%d REPEATS=%d" % (TS, REPEATS), flush=True)

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
        # AABB half-extents: dx_max=sqrt(T*s11), dy_max=sqrt(T*s22), T=gthr
        Tpos = torch.clamp(gthr, min=0.0)
        # FIX (2026-10-05): robust, bit-exact AABB half-extents for the per-tile
        # reject. Three failure modes of the naive sqrt(T*cc/det) are handled:
        #  (a) RATIO-clamp bug: for near-singular gaussians det can be <= 0; the
        #      old code clamped the ratio gcc/det to 1e-30, collapsing the AABB
        #      to ~0 and wrongly skipping the gaussian in every tile but its
        #      center (mode3 != mode0, maxdiff>1). Clamp the DETERMINinant.
        #  (b) float32 vs float64 ellipse: the kernel evaluates q in float32 from
        #      the conic; for ill-conditioned conics the float32 ellipse can be
        #      wider than the float64 one. Take max(float64, float32 extent).
        #  (c) kernel-side cancellation: for high-correlation (near-singular)
        #      conics the kernel float32 q = ca*dx^2+2cb*dx*dy+cc*dy^2 can
        #      catastrophically cancel at far pixels, creating spurious blobs no
        #      finite AABB can bound. Never-skip such gaussians (AABB=1e9).
        det64 = (gca.double() * gcc.double() - gcb.double() * gcb.double()).clamp(min=1e-30)
        gdx64 = torch.sqrt(Tpos.double() * (gcc.double() / det64))
        gdy64 = torch.sqrt(Tpos.double() * (gca.double() / det64))
        det32 = gca * gcc - gcb * gcb
        safe32 = torch.isfinite(det32) & (det32 > 0)
        det32c = torch.where(safe32, det32, torch.full_like(det32, 1e-30))
        dx32 = torch.sqrt(torch.clamp(Tpos * (gcc / det32c), min=0.0))
        dy32 = torch.sqrt(torch.clamp(Tpos * (gca / det32c), min=0.0))
        big = torch.full_like(dx32, 1e9)
        dx32 = torch.where(torch.isfinite(dx32), dx32, big)
        dy32 = torch.where(torch.isfinite(dy32), dy32, big)
        gdx = torch.maximum(gdx64, dx32.double()).float()
        gdy = torch.maximum(gdy64, dy32.double()).float()
        cacg = gca * gcc
        corr = torch.where(cacg > 0, det32 / torch.clamp(cacg, min=1e-30), torch.full_like(det32, -1.0))
        bad = (~torch.isfinite(det32)) | ~(corr > float(os.environ.get("AABB_TAU", "1e-6")))
        n_never = int(bad.sum().item())
        gdx = torch.where(bad, torch.full_like(gdx, 1e9), gdx)
        gdy = torch.where(bad, torch.full_like(gdy, 1e9), gdy)
        MARGIN_ABS = 1e-3; MARGIN_REL = float(os.environ.get("AABB_INFLATE", "1e-2"))
        gdx = (gdx * (1.0 + MARGIN_REL) + MARGIN_ABS).contiguous()
        gdy = (gdy * (1.0 + MARGIN_REL) + MARGIN_ABS).contiguous()
        print("[aabbfix] cam=%d n_never_skip=%d frac=%.4f" % (cid, n_never, n_never / max(gca.numel(), 1)), flush=True)
        gcol = col.contiguous()
        t0 = time.time()
        gids, tile_ptr, num_tiles, ntx, nty, total, capped = build_tile_lists(
            mu2d, conic, depth, radii, cam.width, cam.height, args.r_cap)
        W, H = cam.width, cam.height
        tensors = (gmx, gmy, gca, gcb, gcc, gop, gthr, gcol, gdx, gdy, gids, tile_ptr, ntx, nty, W, H)
        out_b = mod.walk(*tensors, 0, TS)[0]
        out_t = mod.walk(*tensors, 3, TS)[0]
        out_p = mod.walk(*tensors, 4, TS)[0]
        if cid == dev_ids[0]:
            out_sk, out_gid = mod.walk(*tensors, 5, TS)
        else:
            out_sk = out_gid = None
        mb, bmin, bmax = time_mode(mod, tensors, 0, REPEATS)
        mt, tmin, tmax = time_mode(mod, tensors, 3, REPEATS)
        mp, pmin, pmax = time_mode(mod, tensors, 4, REPEATS)
        mf, fmin, fmax = time_mode(mod, tensors, 2, REPEATS)
        out_b2 = mod.walk(*tensors, 0, TS)[0]
        det = (out_b - out_b2).abs().max().item()
        maxdiff_t = (out_b - out_t).abs().max().item()
        maxdiff_p = (out_b - out_p).abs().max().item()
        print("[diag] det=%.2e maxdiff_tile=%.2e maxdiff_pix=%.2e" % (
            det, maxdiff_t, maxdiff_p), flush=True)
        if cid == dev_ids[0]:
            d_t = (out_b - out_t).abs()
            am = int(d_t.argmax())
            print("[range] out_b_max=%.4f out_t_max=%.4f out_b_min=%.4f out_t_min=%.4f maxdiff_at=%d out_b=%.4f out_t=%.4f" % (
                out_b.max().item(), out_t.max().item(), out_b.min().item(), out_t.min().item(),
                am, out_b.view(-1)[am].item(), out_t.view(-1)[am].item()), flush=True)
            if out_sk is not None:
                skmax = float(out_sk.max().item())
                skam = int(out_sk.argmax())
                print("[skipdiag] max_skipped_alpha=%.6f at_pix=%d gid=%d (n_skipped>0=%d)" % (
                    skmax, skam, int(out_gid[skam].item()), int((out_sk > 0).sum().item())), flush=True)
        speedup_t = mb / mt
        speedup_p = mb / mp
        cams_out[str(cid)] = dict(
            n_gaussians=int(gidx.numel()), total_occurrences=int(total),
            capped_fraction=capped, num_tiles=int(num_tiles),
            baseline_ms=mb, tile_ms=mt, pix_ms=mp, floor_ms=mf,
            speedup_tile=speedup_t, speedup_pix=speedup_p,
            max_abs_diff_tile=maxdiff_t, max_abs_diff_pix=maxdiff_p,
            baseline_pinned=[bmin, bmax], tile_pinned=[tmin, tmax],
            pix_pinned=[pmin, pmax], floor_pinned=[fmin, fmax],
            build_s=time.time()-t0)
        print("[cam %d] occ=%d base=%.3fms tile=%.3fms (%.3fx) pix=%.3fms (%.3fx) floor=%.3fms md_t=%.2e md_p=%.2e" % (
            cid, total, mb, mt, speedup_t, mp, speedup_p, mf, maxdiff_t, maxdiff_p), flush=True)

    agg = dict(total_occ=sum(v["total_occurrences"] for v in cams_out.values()),
               baseline_ms=sum(v["baseline_ms"] for v in cams_out.values()),
               tile_ms=sum(v["tile_ms"] for v in cams_out.values()),
               pix_ms=sum(v["pix_ms"] for v in cams_out.values()),
               floor_ms=sum(v["floor_ms"] for v in cams_out.values()))
    agg["speedup_tile"] = agg["baseline_ms"] / agg["tile_ms"]
    agg["speedup_pix"] = agg["baseline_ms"] / agg["pix_ms"]
    agg["max_abs_diff_tile"] = max(v["max_abs_diff_tile"] for v in cams_out.values())
    agg["max_abs_diff_pix"] = max(v["max_abs_diff_pix"] for v in cams_out.values())
    out = dict(schema="bes_aabb_probe_v1", scene=args.scene, gpu_uuid=gpu_uuid,
               r_cap=args.r_cap, repeats=REPEATS, cameras=cams_out, agg=agg)
    print("[agg] " + json.dumps(agg), flush=True)
    out_dir = os.environ.get("BES0_OUTPUT_DIR", ".")
    with open(os.path.join(out_dir, "aabb_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] wrote aabb_%s.json" % args.scene, flush=True)


if __name__ == "__main__":
    main()

