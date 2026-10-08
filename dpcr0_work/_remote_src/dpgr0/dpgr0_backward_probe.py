import sys, importlib.util
sys.path.insert(0, "/root/epic05_dpgr0"); sys.path.insert(0, "/root/epic05_dpc0"); sys.path.insert(0, "/root/epic05_n1")
import numpy as np, torch, n1_core as C, dpc0_common as U
gpu=int(sys.argv[1]); cell="room@10000"
torch.cuda.set_device(gpu); torch.manual_seed(0)
scene, it = U.cell_scene_iter(cell); U.bootstrap("c0")
model, ds = U.setup_state(scene, it)
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
C0_BWD = exp.higs_rasterize_backward
DPC_BWD = U.load_dpc().higs_rasterize_backward
SO="/root/.cache/torch_extensions/py310_cu128/dpgr0_gaussian_render_inference_scene_cuda/dpgr0_gaussian_render_inference_scene_cuda.so"
spec=importlib.util.spec_from_file_location("dpgr0_gaussian_render_inference_scene_cuda", SO)
DPGR0=importlib.util.module_from_spec(spec); spec.loader.exec_module(DPGR0)
DPGR0_BWD=DPGR0.higs_rasterize_backward_dpgr0
PN=U.PARAM_NAMES
def det32(src, bwd, label):
    exp.higs_rasterize_backward=bwd
    L,params,img,gt,cam=U.render_loss(model,ds,src)
    torch.autograd.grad(L,params,retain_graph=True,allow_unused=True); model.optimizer.zero_grad(set_to_none=True)
    hashes=[]
    for r in range(32):
        g=torch.autograd.grad(L,params,retain_graph=True,allow_unused=True)
        for p,gr in zip(params,g): p.grad=None if gr is None else gr.detach().clone()
        hashes.append(U.hash_arrays(U.grads_cpu(model,params)))
    print(f"{label} src{src}: unique={len(set(hashes))}", flush=True)
# rank0=cam20 (nondet in phase), rank1=cam59 (det in phase)
for src in [20, 59]:
    det32(src, C0_BWD,   "C0_default")
    det32(src, DPC_BWD,  "DPC0")
    det32(src, DPGR0_BWD,"DPGR0")