import sys, os, importlib.util, json
sys.path.insert(0, "/root/epic05_dpgr0"); sys.path.insert(0, "/root/epic05_dpc0"); sys.path.insert(0, "/root/epic05_n1")
import numpy as np, torch, n1_core as C, dpc0_common as U

gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CELL = "room@10000"; SRC = "19"
torch.cuda.set_device(gpu); torch.manual_seed(0)
exp = U.bootstrap("c0")
DPGR0_SO = ("/root/.cache/torch_extensions/py310_cu128/"
            "dpgr0_gaussian_render_inference_scene_cuda/"
            "dpgr0_gaussian_render_inference_scene_cuda.so")
spec = importlib.util.spec_from_file_location("dpgr0_gaussian_render_inference_scene_cuda", DPGR0_SO)
dpgr0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dpgr0)
sys.modules["dpgr0_gaussian_render_inference_scene_cuda"] = dpgr0
exp.higs_rasterize_backward = dpgr0.higs_rasterize_backward_dpgr0

scene, it = U.cell_scene_iter(CELL)
model, ds = U.setup_state(scene, it)
cam_idx = int(SRC)
L, params, img, gt, cam = U.render_loss(model, ds, cam_idx)
print("render ok; L=", float(L), "n_params=", len(params))

# one backward
g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
finite = all((gr is None) or torch.isfinite(gr).all().item() for gr in g)
print("grad finite:", finite)
gauss = [gr for gr in g if gr is not None]
print("n non-None grads:", len(gauss))
if gauss:
    print("sample grad norm:", float(gauss[0].norm()))

# getters
st = dpgr0.dpgr0_get_stage_times()
stats = dpgr0.dpgr0_get_stats()
print("stage_times [range,gather,reduce] ms:", st)
print("stats [n_isects,n_seg,n_chunked,total_chunks,pstride,N,records,record_bytes]:", stats)
n_isects, n_seg, n_chunked, total_chunks, pstride, N, records, record_bytes = stats
print("record_ratio = records/n_isects =", records/max(n_isects,1))
print("n_direct_seg =", n_seg - n_chunked)
print("SMOKE_OK" if finite else "SMOKE_NONFINITE")

