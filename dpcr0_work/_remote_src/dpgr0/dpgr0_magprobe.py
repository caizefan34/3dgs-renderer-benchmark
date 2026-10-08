import sys, importlib.util
sys.path.insert(0, "/root/epic05_dpgr0"); sys.path.insert(0, "/root/epic05_dpc0"); sys.path.insert(0, "/root/epic05_n1")
import numpy as np, torch, n1_core as C, dpc0_common as U
gpu=0; cell="room@10000"
torch.cuda.set_device(gpu); torch.manual_seed(0)
scene, it = U.cell_scene_iter(cell); U.bootstrap("c0")
model, ds = U.setup_state(scene, it)
src = U.cell_cameras(scene)["sources"][0]
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
DPC_BWD = U.load_dpc().higs_rasterize_backward
SO="/root/.cache/torch_extensions/py310_cu128/dpgr0_gaussian_render_inference_scene_cuda/dpgr0_gaussian_render_inference_scene_cuda.so"
spec=importlib.util.spec_from_file_location("dpgr0_gaussian_render_inference_scene_cuda", SO)
DPGR0=importlib.util.module_from_spec(spec); spec.loader.exec_module(DPGR0)
DPGR0_BWD=DPGR0.higs_rasterize_backward_dpgr0
PN=U.PARAM_NAMES
def flat(bwd):
    exp.higs_rasterize_backward=bwd
    L,params,img,gt,cam=U.render_loss(model,ds,int(src))
    g=torch.autograd.grad(L,params,retain_graph=True,allow_unused=True)
    for p,gr in zip(params,g): p.grad=None if gr is None else gr.detach().clone()
    a=[]
    for n in PN:
        p=getattr(model,n); gg=p.grad
        a.append(np.zeros(p.numel(),np.float64) if gg is None else gg.detach().cpu().numpy().astype(np.float64).ravel())
    return np.concatenate(a)
ref=flat(DPC_BWD)
for name,bwd in [("DPC0",DPC_BWD),("DPGR0",DPGR0_BWD)]:
    runs=[flat(bwd) for _ in range(4)]
    base=runs[0]
    print(name,"pairwise maxabs / relL2 vs run0:")
    for r in range(1,4):
        d=base-runs[r]; maxabs=float(np.abs(d).max()); rel=float(np.linalg.norm(d)/max(np.linalg.norm(base),1e-300))
        print(f"  run{r}: maxabs={maxabs:.3e} relL2={rel:.3e}")
    d=base-ref; print(f"  vs DPC0: maxabs={float(np.abs(d).max()):.3e} relL2={float(np.linalg.norm(d)/max(np.linalg.norm(ref),1e-300)):.3e}")
