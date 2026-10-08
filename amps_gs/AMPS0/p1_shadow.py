"""AMPS0 P1 shadow-state numerical oracle (v2).
Real gradient stream from the pure-PyTorch 3DGS oracle; FP32 vs BF16-stored moments.
Fixes: (a) only views whose render connects to the autograd graph; (b) real 3DGS
opacity/scale regularizers guarantee connectivity; (c) sh_rest gradients supplied as a
documented controlled synthetic supplement (oracle renders SH-DC only).
"""
import sys, torch, json, time
sys.path.insert(0, ".")
from momentgrad_gs.scene import load_scene
from momentgrad_gs.oracle import project_gaussians, rasterize

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
import os as _os
SCENE=_os.environ.get("AMPS_SCENE","room"); N_ITER=512; RSCALE=0.125; TILE=32
BETA1,BETA2,EPS = 0.9,0.999,1e-15
LR = 1e-3
GROUPS=[("means",3),("scales",3),("quats",4),("opacity",1),("sh0",3),("sh_rest",45)]
NAMES=[g[0] for g in GROUPS]; SIZES={g[0]:g[1] for g in GROUPS}

def bf16(x): return x.to(torch.bfloat16).to(torch.float32)

cams, init = load_scene(f"data/datasets/mipnerf360/{SCENE}", split="train", every=8, max_points=400, device=device)
print(f"loaded {len(cams)} cams, N={init.means.shape[0]}", flush=True)

def render(p, cam):
    H=max(16,int(cam.height*RSCALE)); W=max(16,int(cam.width*RSCALE))
    proj=project_gaussians(p["means"],p["quats"],p["scales"],torch.sigmoid(p["opacity"]),
        cam.worldtocam,cam.K,colors_dc=p["sh0"],max_radius=256)
    return rasterize(proj,H,W,pixel_loss_grad=torch.ones(H,W,3,device=device),tile=TILE).image

# reference target from perturbed params
torch.manual_seed(1)
pert={k:v+0.02*torch.randn_like(v) for k,v in init.__dict__.items() if hasattr(v,'shape') and isinstance(v,torch.Tensor)} if False else None
p0={"means":init.means.clone(),"scales":init.scales.clone(),"quats":init.quats.clone(),
    "opacity":init.opacities.clone(),"sh0":init.colors.clone()}
with torch.no_grad(): target=render({k:v.clone() for k,v in p0.items()}, cams[0])

# base buffer for jitter; fresh leaves each iter
base={k:v.clone() for k,v in p0.items()}
# synthetic sh_rest gradient generator (documented): higher-order SH grads are smaller & sparser
g_shrest = (torch.randn(N_ITER, SIZES["sh_rest"], device=device) * 0.05)
g_shrest = g_shrest * (torch.rand_like(g_shrest) > 0.3)  # ~70% sparse

# pre-filter VALID views (render connects to autograd graph) once
print("pre-filtering valid views...", flush=True)
valid_cams=[]
for ci,cam in enumerate(cams):
    pc={k:base[k].clone().detach().requires_grad_(True) for k in base}
    img=render(pc,cam)
    if img.requires_grad:
        valid_cams.append(cam)
    if len(valid_cams)>=40: break
print(f"valid views: {len(valid_cams)}", flush=True)
assert len(valid_cams)>0, "no valid views found"

grad_stream=[]
t0=time.time(); it=0
while it < N_ITER:
    cam=valid_cams[it % len(valid_cams)]
    pc={k:base[k].clone().detach().requires_grad_(True) for k in base}
    img=render(pc,cam)
    if not img.requires_grad:   # safety: should not happen post-filter
        it+=1; continue
    a=img.reshape(img.shape[0],-1); b=target.reshape(target.shape[0],-1)
    cos=(a*b).sum(1)/((a.norm(2,1)*b.norm(2,1))+1e-30)
    # real 3DGS regularizers: opacity + scale (also guarantee connectivity)
    reg = 1e-4*torch.sigmoid(pc["opacity"]).mean() + 1e-4*torch.exp(pc["scales"]).mean()
    loss = torch.nn.functional.l1_loss(img,target) + 0.5*(1-cos).mean() + reg
    leaves=[pc[k] for k in ["means","scales","quats","opacity","sh0"]]
    g=torch.autograd.grad(loss, leaves)
    step={k:gi.detach().clone() for k,gi in zip(["means","scales","quats","opacity","sh0"], g)}
    step["sh_rest"]=g_shrest[it].detach().clone()
    grad_stream.append(step)
    with torch.no_grad():
        for k in base: base[k].add_(0.001*torch.randn_like(base[k]))
    it+=1
print(f"generated {len(grad_stream)} real grad steps in {time.time()-t0:.1f}s", flush=True)
import os as _os
_cache=f"amps_gs/AMPS0/raw/grad_stream_{SCENE}.pt"
if not _os.path.exists(_cache):
    torch.save(grad_stream, _cache); print("cached grad_stream", flush=True)

def run_shadow(policy):
    m={k:torch.zeros_like(grad_stream[0][k].reshape(-1)) for k in NAMES}
    v={k:torch.zeros_like(grad_stream[0][k].reshape(-1)) for k in NAMES}
    m_dt={k:policy.get(k,{}).get('m','fp32') for k in NAMES}
    v_dt={k:policy.get(k,{}).get('v','fp32') for k in NAMES}
    upd={k:[] for k in NAMES}
    for t,gs in enumerate(grad_stream):
        for k in NAMES:
            g=gs[k].reshape(-1)
            m[k]=BETA1*m[k]+(1-BETA1)*g
            v[k]=BETA2*v[k]+(1-BETA2)*g*g
            mh=m[k]/(1-BETA1**(t+1)); vh=v[k]/(1-BETA2**(t+1))
            upd[k].append((LR*mh/(torch.sqrt(vh)+EPS)).detach())
            m[k]=bf16(m[k]) if m_dt[k]=='bf16' else m[k]
            v[k]=bf16(v[k]) if v_dt[k]=='bf16' else v[k]
    return {k:torch.stack(upd[k]) for k in NAMES}

t0=time.time(); ref=run_shadow({k:{} for k in NAMES}); print(f"FP32 shadow {time.time()-t0:.1f}s", flush=True)

policies={"C1":{"sh_rest":{"m":"fp32","v":"bf16"}},
          "C2":{"sh_rest":{"m":"bf16","v":"bf16"}},
          "C3":{"sh_rest":{"m":"bf16","v":"bf16"},"sh0":{"m":"bf16","v":"bf16"},"opacity":{"m":"bf16","v":"bf16"}},
          "C4":{k:{"m":"bf16","v":"bf16"} for k in NAMES}}

def metrics(ru,cu):
    dot=(ru*cu).sum(1); nr=ru.norm(2,1); nc=cu.norm(2,1)
    cos=dot/((nr*nc)+1e-30)
    rel=(cu-ru).norm(2,1)/((nr)+1e-30)
    maxa=(cu-ru).abs().max(1).values
    thr=0.01*nr[:,None]; sig=ru.abs()>thr
    srate=(((ru*cu)<0)&sig).sum(1).float()/sig.sum(1).clamp_min(1)
    return {"cosine_median":float(cos.median()),"cosine_p01":float(torch.quantile(cos,0.01)),
            "cosine_p50":float(torch.quantile(cos,0.50)),"cosine_p90":float(torch.quantile(cos,0.90)),
            "rel_l2_median":float(rel.median()),"rel_l2_p95":float(torch.quantile(rel,0.95)),
            "max_abs_median":float(maxa.median()),"sign_disagree_median":float(srate.median()),
            "finite":bool(torch.isfinite(cu).all())}

res={"schema":"AMPS0_SHADOW_NUMERICS","generated":time.strftime("%Y-%m-%d"),"scene":SCENE,
     "n_iter":len(grad_stream),"device":device,"rscale":RSCALE,"n_gaussians":int(init.means.shape[0]),
     "beta1":BETA1,"beta2":BETA2,"eps":EPS,
     "gradient_source":"real pure-PyTorch oracle render (room, COLMAP init); params frozen at FP32; only moments differ. sh_rest = documented synthetic supplement (oracle renders SH-DC only). Views that cull all Gaussians are skipped.",
     "note":"Pure state-storage error isolation, NOT a trajectory claim. BF16 = round-to-nearest-even via torch.bfloat16.",
     "candidates":{}}
res["candidates"]["C0"]={"policy":"all FP32","per_group":{k:metrics(ref[k],ref[k]) for k in NAMES}}
for cn,pol in policies.items():
    cand=run_shadow(pol)
    res["candidates"][cn]={"policy":pol,"per_group":{k:metrics(ref[k],cand[k]) for k in NAMES}}
    print(f"{cn} done", flush=True)

def elig(c,g):
    m=c["per_group"][g]
    return (m["cosine_median"]>=0.9999 and m["cosine_p01"]>=0.999 and m["rel_l2_median"]<=5e-3
            and m["rel_l2_p95"]<=2e-2 and m["finite"] and m["sign_disagree_median"]<=0.005)
eg={}
for cn in ["C1","C2","C3","C4"]:
    c=res["candidates"][cn]; mod=list(c["policy"].keys())
    eg[cn]={g:elig(c,g) for g in mod}; eg[cn]["ALL_PASS"]=all(eg[cn].values())
res["eligibility_gate"]=eg
json.dump(res, open(f"amps_gs/AMPS0/AMPS0_SHADOW_NUMERICS_{SCENE}.json","w"), indent=2)
print("\n=== SUMMARY (sh_rest) ===")
for cn in ["C0","C1","C2","C3","C4"]:
    m=res["candidates"][cn]["per_group"]["sh_rest"]
    print(cn,"cos_med=%.5f relL2_med=%.2e relL2_p95=%.2e sign=%.4f fin=%s"%(m["cosine_median"],m["rel_l2_median"],m["rel_l2_p95"],m["sign_disagree_median"],m["finite"]))
print("eligibility:", json.dumps(eg))
print("P1_DONE")
