import sys, importlib.util
sys.path.insert(0, "/root/epic05_dpgr0"); sys.path.insert(0, "/root/epic05_dpc0"); sys.path.insert(0, "/root/epic05_n1")
import numpy as np, torch, n1_core as C, dpc0_common as U
gpu=int(sys.argv[1]); cell="room@10000"; src=20; NREP=int(sys.argv[2]) if len(sys.argv)>2 else 32
torch.cuda.set_device(gpu); torch.manual_seed(0)
scene, it = U.cell_scene_iter(cell); U.bootstrap("c0")
model, ds = U.setup_state(scene, it)
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
SO="/root/.cache/torch_extensions/py310_cu128/dpgr0_gaussian_render_inference_scene_cuda/dpgr0_gaussian_render_inference_scene_cuda.so"
spec=importlib.util.spec_from_file_location("dpgr0_gaussian_render_inference_scene_cuda", SO)
DPGR0=importlib.util.module_from_spec(spec); spec.loader.exec_module(DPGR0)
exp.higs_rasterize_backward=DPGR0.higs_rasterize_backward_dpgr0
PN=U.PARAM_NAMES
# EXACT phase det32 mirror
L,params,img,gt,cam=U.render_loss(model,ds,src)
torch.autograd.grad(L,params,retain_graph=True,allow_unused=True); model.optimizer.zero_grad(set_to_none=True)
hashes=[]
for r in range(NREP):
    grads=torch.autograd.grad(L,params,retain_graph=True,allow_unused=True)
    for p,gr in zip(params,grads): p.grad=None if gr is None else gr.detach().clone()
    arrs=U.grads_cpu(model,params)
    hashes.append(U.hash_arrays(arrs))
uniq=sorted(set(hashes))
print(f"MIRROR det{NREP} source{src} gpu{gpu} cell={cell}: unique={len(uniq)}")
if len(uniq)>1:
    base=hashes[0]
    for r in range(NREP):
        if hashes[r]!=base: print(f"  MISMATCH rep{r}: {hashes[r]}")