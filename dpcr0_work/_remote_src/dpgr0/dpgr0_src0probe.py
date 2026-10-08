import sys, importlib.util
sys.path.insert(0, "/root/epic05_dpgr0"); sys.path.insert(0, "/root/epic05_dpc0"); sys.path.insert(0, "/root/epic05_n1")
import numpy as np, torch, n1_core as C, dpc0_common as U
gpu=0; cell="room@10000"; src=20
torch.cuda.set_device(gpu); torch.manual_seed(0)
scene, it = U.cell_scene_iter(cell); U.bootstrap("c0")
model, ds = U.setup_state(scene, it)
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
DPC_BWD = U.load_dpc().higs_rasterize_backward
SO="/root/.cache/torch_extensions/py310_cu128/dpgr0_gaussian_render_inference_scene_cuda/dpgr0_gaussian_render_inference_scene_cuda.so"
spec=importlib.util.spec_from_file_location("dpgr0_gaussian_render_inference_scene_cuda", SO)
DPGR0=importlib.util.module_from_spec(spec); spec.loader.exec_module(DPGR0)
DPGR0_BWD=DPGR0.higs_rasterize_backward_dpgr0
PN=U.PARAM_NAMES
def grads():
    a=[]
    for n in PN:
        p=getattr(model,n); g=p.grad
        a.append(np.zeros(p.numel(),np.float64) if g is None else g.detach().cpu().numpy().astype(np.float64).ravel())
    return np.concatenate(a)
def setg(grads):
    for p,gr in zip([getattr(model,n) for n in PN],grads):
        p.grad = None if gr is None else torch.as_tensor(gr,dtype=torch.float32,device=p.device).reshape(p.shape)
# Method A: retain_graph, one render, 8 reps
exp.higs_rasterize_backward=DPGR0_BWD
L,params,img,gt,cam=U.render_loss(model,ds,src)
torch.autograd.grad(L,params,retain_graph=True,allow_unused=True); model.optimizer.zero_grad(set_to_none=True)
runsA=[]
for r in range(8):
    g=torch.autograd.grad(L,params,retain_graph=True,allow_unused=True)
    for p,gr in zip(params,g): p.grad=None if gr is None else gr.detach().clone()
    runsA.append(grads()); model.optimizer.zero_grad(set_to_none=True)
hA=[U.hash_arrays([r]) for r in runsA]
print("A retain_graph: unique=",len(set(hA)))
base=runsA[0]
for r in range(1,8):
    d=base-runsA[r]; print(f"  A run{r}: maxabs={float(np.abs(d).max()):.3e} relL2={float(np.linalg.norm(d)/max(np.linalg.norm(base),1e-300)):.3e}")
# Method B: fresh render each rep, 8 reps
runsB=[]
for r in range(8):
    L,params,img,gt,cam=U.render_loss(model,ds,src)
    g=torch.autograd.grad(L,params,retain_graph=False,allow_unused=True)
    for p,gr in zip(params,g): p.grad=None if gr is None else gr.detach().clone()
    runsB.append(grads()); model.optimizer.zero_grad(set_to_none=True)
hB=[U.hash_arrays([r]) for r in runsB]
print("B fresh_render: unique=",len(set(hB)))
base=runsB[0]
for r in range(1,8):
    d=base-runsB[r]; print(f"  B run{r}: maxabs={float(np.abs(d).max()):.3e} relL2={float(np.linalg.norm(d)/max(np.linalg.norm(base),1e-300)):.3e}")
