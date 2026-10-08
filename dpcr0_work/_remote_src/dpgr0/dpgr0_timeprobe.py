import sys, importlib.util
sys.path.insert(0, "/root/epic05_dpgr0"); sys.path.insert(0, "/root/epic05_dpc0"); sys.path.insert(0, "/root/epic05_n1")
import numpy as np, torch, n1_core as C, dpc0_common as U

gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cell = sys.argv[2] if len(sys.argv) > 2 else "room@10000"
torch.cuda.set_device(gpu); torch.manual_seed(0)
scene, it = U.cell_scene_iter(cell)
U.bootstrap("c0")
model, ds = U.setup_state(scene, it)
scene, _it = U.cell_scene_iter(cell); cc = U.cell_cameras(scene); src = cc["sources"][0]
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
C0_BWD = exp.higs_rasterize_backward
DPC_BWD = U.load_dpc().higs_rasterize_backward
DPGR0_SO = ("/root/.cache/torch_extensions/py310_cu128/"
            "dpgr0_gaussian_render_inference_scene_cuda/dpgr0_gaussian_render_inference_scene_cuda.so")
spec = importlib.util.spec_from_file_location("dpgr0_gaussian_render_inference_scene_cuda", DPGR0_SO)
DPGR0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(DPGR0)

def time_bwd(bwd, n=30):
    exp.higs_rasterize_backward = bwd
    L, params, img, gt, cam = U.render_loss(model, ds, int(src))
    for _ in range(8):
        torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
        model.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        f0 = torch.cuda.Event(enable_timing=True); f1 = torch.cuda.Event(enable_timing=True)
        f0.record()
        torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
        f1.record(); torch.cuda.synchronize()
        ts.append(f0.elapsed_time(f1))
        model.optimizer.zero_grad(set_to_none=True)
    return float(np.median(ts)), float(np.percentile(ts, 90))

for name, bwd in [("C0", C0_BWD), ("DPC0", DPC_BWD), ("DPGR0", DPGR0.higs_rasterize_backward_dpgr0)]:
    med, p90 = time_bwd(bwd)
    print(f"{name}: median={med:.3f} ms  p90={p90:.3f} ms")

