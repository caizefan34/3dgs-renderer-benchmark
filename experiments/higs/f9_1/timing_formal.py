"""Authoritative composed-path F9-1R timing: 20 warmup, 5x100 interleaved."""
import json, os, sys
import numpy as np
import torch
from f9_0_loader import load_scene
from gsplat.experimental.render.functional.gaussian_inference import _higs_dynamic_forward

scene, out_path = sys.argv[1:]
m,q,s,o,sh,vm,K,w,h,_ = load_scene(scene)
for x in (m,q,s,o,sh): x.requires_grad_(True)

def invoke(disable, backward=False):
    os.environ["HIGS_DISABLE_F9"] = "1" if disable else "0"
    r = _higs_dynamic_forward(m,q,s,o,sh, viewmats=vm,Ks=K,width=w,height=h,
        sh_degree=3,backward_mode="higs_native",enable_culling=True,
        camera_model="pinhole",render_mode="RGB",eps2d=.3)
    if backward:
        (r["frame"] * noise_f).sum().add((r["alpha"] * noise_a).sum()).backward()
    return r

def clear_grads():
    for x in (m,q,s,o,sh): x.grad = None
def event(fn):
    torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True)
    a.record(); fn(); b.record(); b.synchronize(); return a.elapsed_time(b)

# Fix upstream gradients before warmup; they are never included in event timing.
r = invoke(True)
noise_f=torch.randn_like(r["frame"]); noise_a=torch.randn_like(r["alpha"]); clear_grads()
# Warm both paths, then measure the same composed paths.
for _ in range(20):
    invoke(True); invoke(False); clear_grads(); invoke(True,True); clear_grads(); invoke(False,True); clear_grads()
samples={k:[] for k in ("baseline_forward","f9_forward","baseline_fb","f9_fb")}
for rep in range(5):
    for _ in range(100):
        samples["baseline_forward"].append(event(lambda:invoke(True)))
        samples["f9_forward"].append(event(lambda:invoke(False)))
        clear_grads(); samples["baseline_fb"].append(event(lambda:invoke(True,True))); clear_grads()
        samples["f9_fb"].append(event(lambda:invoke(False,True))); clear_grads()

def stat(x):
    x=np.asarray(x,float); rng=np.random.default_rng(20260921); boots=np.array([np.median(rng.choice(x,len(x),replace=True)) for _ in range(2000)])
    return {"median_ms":float(np.median(x)),"mean_ms":float(np.mean(x)),"p10_ms":float(np.percentile(x,10)),"p90_ms":float(np.percentile(x,90)),"std_ms":float(np.std(x)),"bootstrap_median_ci95_ms":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],"n":len(x)}
json.dump({"scene":scene,"resolution":[w,h],"protocol":{"warmup":20,"samples_per_repetition":100,"repetitions":5,"interleaved":True},"timing":{k:stat(v) for k,v in samples.items()}},open(out_path,"w"),indent=2)
