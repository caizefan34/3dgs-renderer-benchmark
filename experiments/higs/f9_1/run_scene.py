import json, os, sys
import torch
from experiments.higs.f9_0.run_f9_0 import load_scene
from gsplat.experimental.render.functional.gaussian_inference import _higs_dynamic_forward

scene = sys.argv[1]
out_path = sys.argv[2]
base = load_scene(scene)
m0, q0, s0, o0, sh0, vm, K, w, h, image = base
torch.manual_seed(20260921)

def leaf(x): return x.detach().clone().requires_grad_(True)
def run(disable=False, backward=True):
    m,q,s,o,sh = map(leaf, (m0,q0,s0,o0,sh0))
    os.environ["HIGS_DISABLE_F9"] = "1" if disable else "0"
    result = _higs_dynamic_forward(m,q,s,o,sh, viewmats=vm, Ks=K, width=w, height=h,
        sh_degree=3, backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB", eps2d=.3)
    frame, alpha = result["frame"], result["alpha"]
    noise_f = torch.randn_like(frame)
    noise_a = torch.randn_like(alpha)
    loss = (frame * noise_f).sum() + (alpha * noise_a).sum()
    if backward: loss.backward()
    return result, (m,q,s,o,sh), noise_f, noise_a

def metric(a,b):
    a,b=a.float(),b.float(); d=a-b
    return {"max_abs":float(d.abs().max()), "rel_l2":float(d.norm()/b.norm().clamp_min(1e-30)),
        "cosine":float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))}

# Correctness fresh parameter copies, identical upstream noise by rerunning with
# the same RNG state after the path-independent scene load.
torch.manual_seed(73); b,bg,bnf,bna=run(True)
torch.manual_seed(73); f,fg,fnf,fna=run(False)
bm=b["metadata"]; fm=f["metadata"]
correct={"frame":metric(f["frame"],b["frame"]), "alpha":metric(f["alpha"],b["alpha"]),
    "grads":{k:metric(x.grad,y.grad) for k,x,y in zip(["means","quats","scales","opacities","sh"],fg,bg)},
    "n_isects":[bm["n_isects"],fm["n_isects"]], "visible":[bm["n_visible"],fm["n_visible"]],
    "f9_enabled":fm.get("f9_enabled",False)}

def event(fn):
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); fn(); b.record(); b.synchronize(); return a.elapsed_time(b)
def step(disable, bwd):
    def call():
        r,g,nf,na=run(disable,bwd)
        return r
    return call

for _ in range(5): step(True,False)(); step(False,False)()
times={"base_forward":[],"f9_forward":[],"base_fb":[],"f9_fb":[]}
for _ in range(20):
    times["base_forward"].append(event(step(True,False)))
    times["f9_forward"].append(event(step(False,False)))
    times["base_fb"].append(event(step(True,True)))
    times["f9_fb"].append(event(step(False,True)))
def stats(x):
    t=torch.tensor(x,dtype=torch.float64)
    return {"median_ms":float(t.median()),"mean_ms":float(t.mean()),"p10_ms":float(t.quantile(.1)),"p90_ms":float(t.quantile(.9)),"std_ms":float(t.std(unbiased=False)),"samples_ms":x}
json.dump({"scene":scene,"resolution":[w,h],"n_total":len(m0),"correctness":correct,
    "timing":{k:stats(v) for k,v in times.items()}},open(out_path,"w"),indent=2)
