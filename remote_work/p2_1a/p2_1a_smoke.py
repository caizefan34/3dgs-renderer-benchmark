#!/usr/bin/env python3
"""P2-1A runtime entry smoke matrix.

Modes:
  render   : GaussianInferenceRenderer(...).render(...)          (stateful scene render)
  op       : torch.ops.experimental.gaussian_render_inference_only(...)  (stateless op)
  producer : higs_gatherless_projected_producer(...)             (F9 FP32 producer binding)

Run one (so, mode) pair per process: TEST and C0 .so both register
TORCH_LIBRARY(experimental) and cannot coexist in one process.

No timing. This only answers: which entry points of the built artifact execute?
"""
import sys, os, argparse, importlib.util, traceback
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--so', required=True)
ap.add_argument('--tag', default='run')
ap.add_argument('--mode', default='render', choices=['render', 'op', 'producer'])
ap.add_argument('--n', type=int, default=64)
args = ap.parse_args()

torch.manual_seed(0)
dev = torch.device('cuda')
print('[%s/%s] torch %s, cuda %s, dev %s' % (args.tag, args.mode, torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)), flush=True)

spec = importlib.util.spec_from_file_location('experimental_gaussian_render_inference_scene_cuda', args.so)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

N = args.n
means = (torch.rand(N, 3, device=dev) * 2 - 1).float()
quats = torch.randn(N, 4, device=dev)
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = (torch.rand(N, 3, device=dev) * 0.1 + 0.01).float()
opac = torch.rand(N, device=dev).float()
sh = (torch.randn(N, 16, 3, device=dev) * 0.1).float()

means_planar = means.t().contiguous()                                              # [3,N] float
qso_packed = torch.cat([quats, scales, opac[:, None]], dim=1).half().contiguous()  # [N,8] half
colors_packed = sh.half().contiguous()                                             # [N,16,3] half

vm = torch.eye(4, device=dev)
vm[2, 3] = 5.0                                                                    # camera at (0,0,-5) looking +z
W, H = 256, 192
K = torch.tensor([[200.0, 0.0, (W - 1) / 2],
                  [0.0, 200.0, (H - 1) / 2],
                  [0.0, 0.0, 1.0]], device=dev)

try:
    if args.mode == 'render':
        r = mod.GaussianInferenceRenderer(means_planar, qso_packed, colors_packed, 3, 0)
        print('[%s/render] renderer created, num_gaussians=%d' % (args.tag, r.num_gaussians()), flush=True)
        out = r.render(means_planar, qso_packed, colors_packed, vm, K, W, H,
                       16, 0.01, 1e10, 0.0, 0.3, 3, 0, None, None)
        torch.cuda.synchronize()
        print('[%s/render] OK shape=%s dtype=%s rgb_mean=%.6f T_mean=%.6f' % (
            args.tag, tuple(out.shape), out.dtype,
            out[..., :3].float().mean().item(), out[..., 3].float().mean().item()), flush=True)
    elif args.mode == 'op':
        renders, alphas = torch.ops.experimental.gaussian_render_inference_only(
            means_planar, qso_packed, colors_packed, vm, K, W, H, 3, 16,
            0.01, 1e10, 0.0, 0.3, 0, None)
        torch.cuda.synchronize()
        print('[%s/op] OK renders=%s dtype=%s alphas=%s dtype=%s' % (
            args.tag, tuple(renders.shape), renders.dtype, tuple(alphas.shape), alphas.dtype), flush=True)
    else:  # producer
        cam_pos = mod.higs_camera_positions_from_viewmats(vm[None].contiguous())
        vis_ids = torch.arange(N, dtype=torch.int64, device=dev)
        radii, means2d, depths, conics, opacities_bc, colors_eval = (
            mod.higs_gatherless_projected_producer(
                vis_ids, means.contiguous(), quats.contiguous(), scales.contiguous(),
                opac.contiguous(), sh.contiguous(), vm[None].contiguous(), K[None].contiguous(),
                cam_pos, W, H, 0.3, 0.01, 1e10, 0.0))
        torch.cuda.synchronize()
        n_vis = int((radii.reshape(-1, 2)[:, 0] > 0).sum())
        print('[%s/producer] OK radii=%s means2d=%s depths=%s conics=%s(%s) opac=%s colors=%s(%s) n_positive_radii=%d' % (
            args.tag, tuple(radii.shape), tuple(means2d.shape), tuple(depths.shape),
            tuple(conics.shape), conics.dtype, tuple(opacities_bc.shape),
            tuple(colors_eval.shape), colors_eval.dtype, n_vis), flush=True)
    print('[%s/%s] SMOKE_RESULT: PASS' % (args.tag, args.mode), flush=True)
except Exception as e:
    print('[%s/%s] FAILED: %s: %s' % (args.tag, args.mode, type(e).__name__, e), flush=True)
    traceback.print_exc()
    print('[%s/%s] SMOKE_RESULT: FAIL' % (args.tag, args.mode), flush=True)
    sys.exit(3)
