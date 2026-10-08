#!/usr/bin/env python3
import sys, os, time
sys.path.insert(0, "/root/hm0_pkg")
sys.path.insert(0, "/root/epic05_moonshot/a0")
import hm0_common as C
import a0_lib as A
import torch
import torch.nn.functional as F
env = C.bootstrap_c0()
from gsplat.experimental.render.functional.gaussian_inference import (
    rasterize_gaussian_higs_dynamic, _HigsDynamicScene)

def sync(): torch.cuda.synchronize()
def isolated_render(means, quats, scales, opac, colors, cams, sh):
    viewmats = torch.stack([c.viewmatrix for c in cams]).unsqueeze(0)
    Ks = torch.stack([c.K for c in cams]).unsqueeze(0)
    H, W = cams[0].image_height, cams[0].image_width
    bg = torch.zeros(3, device="cuda")
    scene = _HigsDynamicScene()
    out = rasterize_gaussian_higs_dynamic(
        means, quats, scales, opac, colors, sh_degree=sh,
        viewmats=viewmats, Ks=Ks, width=W, height=H, background=bg,
        render_mode="RGB", enable_culling=True, camera_model="pinhole",
        backward_mode="higs_native", eps2d=0.3, scene=scene)
    return out["frame"], out["alpha"]
def activated(g):
    return (g["means"], F.normalize(g["quats"],dim=-1), torch.exp(g["scales"]),
            torch.sigmoid(g["opac"]), g["colors"])
def summary(frame, alpha): return frame.sum() + alpha.sum()
def tangents(g):
    T = {}
    for ax in (0,1,2):
        t={k:torch.zeros_like(g[k]) for k in g}; t["means"][:,ax]=1.0; T[f"trans_{ax}"]=t
    for ax in (0,1,2):
        t={k:torch.zeros_like(g[k]) for k in g}; t["scales"][:,ax]=1.0; T[f"logscale_{ax}"]=t
    for ax in (0,1,2):
        t={k:torch.zeros_like(g[k]) for k in g}; p=torch.zeros_like(g["quats"]); p[:,ax+1]=1.0; t["quats"]=0.5*p; T[f"rot_{ax}"]=t
    t={k:torch.zeros_like(g[k]) for k in g}; t["opac"][:]=1.0; T["opacity"]=t
    t={k:torch.zeros_like(g[k]) for k in g}; t["colors"][:,0,:]=1.0; T["sh_intensity"]=t
    t={k:torch.zeros_like(g[k]) for k in g}; t["colors"][:,0,:]=torch.tensor([1.0,-0.5,-0.5],device="cuda"); T["sh_chroma"]=t
    return T
def resp_grad(leaves, cams, sh, T):
    g={k:v.detach().requires_grad_(True) for k,v in leaves.items()}
    me,qu,sc,op,co=activated(g)
    fr,al=isolated_render(me,qu,sc,op,co,cams,sh)
    S=summary(fr,al)
    grads=torch.autograd.grad(S,list(g.values()),create_graph=True)
    gv=dict(zip(g.keys(),grads))
    return torch.tensor([float(sum((gv[k]*t[k]).sum() for k in g)) for t in T.values()])

scene,it="room",10000
ds=C.load_dataset(scene,device="cuda")
cam_ids=C.frozen_camera_ids(len(ds))
m,opt_state,ident,osd,slrs=C.load_model(scene,it,device="cuda")
sh=min(ident["active_sh_degree"],C.SH_DEGREE); N=ident["num_points"]
cams=[ds.get_camera(ci) for ci in cam_ids]
# build one K=16 group via cdist on a subset neighborhood (use first visible seed)
info_uni,uninstall=A.install_capture_patch(); A.A0_CAPTURE["enabled"]=True
isect=torch.zeros(N,device="cuda")
for ci in cam_ids:
    A.A0_CAPTURE["records"].clear()
    cam,gt=ds.get_item(ci); frame,info,out=C.render_c0(m,cam,sh); C.train_loss(frame,gt,C.LAMBDA_DSSIM).backward()
    cap=A.parse_capture(A.A0_CAPTURE["records"][-1])
    isect+=torch.bincount(cap["visible_ids"][cap["flatten_ids_f"].reshape(-1)],minlength=N).float()
uninstall(); sync()
# brute-force knn for one seed using a 60k visible subsample (timing test only)
vis=(isect>0).nonzero().reshape(-1)
sub=vis[:60000]
seed=int(vis[torch.argsort(isect[vis],descending=True)[0]])
with torch.no_grad():
    d=torch.cdist(m["xyz"][seed:seed+1],m["xyz"][sub]).reshape(-1)
    ii=torch.topk(d,17,largest=False).indices[1:]
    gidx=sub[ii]
K=16
g={"means":m["xyz"][gidx].detach().clone().float().requires_grad_(True),
   "quats":m["rotation"][gidx].detach().clone().float().requires_grad_(True),
   "scales":m["scaling"][gidx].detach().clone().float().requires_grad_(True),
   "opac":m["opacity"][gidx].detach().clone().float().requires_grad_(True),
   "colors":m["shs"][gidx].detach().clone().float().requires_grad_(True)}
T=tangents(g)
# group target (fixed)
t0=time.perf_counter()
me,qu,sc,op,co=activated(g); fr_g,al_g=isolated_render(me,qu,sc,op,co,cams,sh)
with torch.no_grad(): Rg=resp_grad({k:v.detach() for k,v in g.items()},cams,sh,T)
sync(); print(f"group target (render+resp): {1000*(time.perf_counter()-t0):.0f} ms  Rg={[round(x,3) for x in Rg.tolist()]}",flush=True)

# macro moment init (fixed key opac)
def macro_init(g):
    with torch.no_grad():
        me,qu,sc,op,co=activated(g)
        w=op.clamp(1e-4,1-1e-4)
        M=(w[:,None]*me).sum(0)/w.sum(); dm=me-M
        cov=(w[:,None,None]*(dm[:,:,None]*dm[:,None,:])).sum(0)/w.sum()
        ev,evec=torch.linalg.eigh(cov); scale=torch.sqrt(ev.clamp(min=1e-10)); Rm=evec.T
        tr=Rm[0,0]+Rm[1,1]+Rm[2,2]; q0=torch.sqrt(torch.clamp(1+tr,0))/2
        q=torch.stack([q0,(Rm[2,1]-Rm[1,2])/(4*q0+1e-12),(Rm[0,2]-Rm[2,0])/(4*q0+1e-12),(Rm[1,0]-Rm[0,1])/(4*q0+1e-12)])
        q=F.normalize(q,dim=-1); opm=float(torch.sigmoid(g["opac"]).mean()); com=co.mean(0,keepdim=True)
        return dict(means=M.clone().requires_grad_(True),quats=q.clone().requires_grad_(True),
                    scales=torch.log(scale).clone().requires_grad_(True),
                    opac=torch.tensor(float(torch.logit(torch.tensor(opm))),device="cuda").requires_grad_(True),
                    colors=com.clone().requires_grad_(True))
mp=macro_init(g)
opt=torch.optim.Adam([mp["means"],mp["quats"],mp["scales"],mp["opac"],mp["colors"]],lr=0.02)
t0=time.perf_counter()
for step in range(20):
    opt.zero_grad()
    me,qu,sc,op,co=activated(mp)
    fr,al=isolated_render(me,qu,sc,op,co,cams,sh)
    loss=(fr-fr_g).pow(2).sum()+(al-al_g).pow(2).sum()
    Rm=resp_grad(mp,cams,sh,T)
    lossloss=(Rm-Rg.detach()).pow(2).sum()
    (loss+lossloss).backward()
    opt.step()
sync()
print(f"macro fit 20 steps (with double-backward resp): {time.perf_counter()-t0:.2f}s ({1000*(time.perf_counter()-t0)/20:.0f} ms/step)",flush=True)
print("RESP_TEST_DONE",flush=True)