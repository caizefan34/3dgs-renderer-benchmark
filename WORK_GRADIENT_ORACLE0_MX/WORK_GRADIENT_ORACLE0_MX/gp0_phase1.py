#!/usr/bin/env python3
"""GRADIENT-PROXY-P0 Phase 1 orchestrator: subprocess-per-render to avoid CUDA corruption."""
import os, sys, json, subprocess, numpy as np, math
D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT=f"{D}/s1a_r1_deliverables"; IMGDIR=f"{OUT}/gp0_renders"; os.makedirs(IMGDIR,exist_ok=True)
CKPTS=["5K","10K","15K"]; CAMS=[0,1]; PY="/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python"
env=f"PYTHONNOUSERSITE=1 TMPDIR={D} TORCH_EXTENSIONS_DIR={D}/torch_ext XDG_CACHE_HOME={D}/xdg"

def run(which, ct, ci, mode, timeout=900):
    cmd=f"cd {D} && {env} {PY} {D}/gp0_phase1_worker.py {which} {ct} {ci} {IMGDIR} {mode} 2>&1"
    r=subprocess.run(["bash","-c",cmd],capture_output=True,text=True,timeout=timeout)
    meta=None
    for line in r.stdout.splitlines():
        if line.startswith("META:"): meta=json.loads(line[5:])
    err=r.stderr[-600:] if r.returncode!=0 else ""
    return r.returncode, meta, err

results={}
for ct in CKPTS:
    for ci in CAMS:
        for which in ["B","C"]:
            rc,meta,err=run(which,ct,ci,"render")
            results[f"{ct}_{which}_cam{ci}_render"]={"rc":rc,"meta":meta,"err":err}
            print(f"[{ct} {which} cam{ci} render] rc={rc} psnr={meta.get('psnr_vs_gt') if meta else 'NA'} {('ERR:'+err) if err else ''}",flush=True)
        for which in ["B","C"]:
            rc,meta,err=run(which,ct,ci,"grad")
            results[f"{ct}_{which}_cam{ci}_grad"]={"rc":rc,"meta":meta,"err":err}
            print(f"[{ct} {which} cam{ci} grad] rc={rc} {('ERR:'+err) if err else ''}",flush=True)

# aggregate: B vs C render agreement (per ckpt,cam), gradient cosine (per ckpt,cam)
def psnr_np(a,b):
    mse=np.square(np.asarray(a,np.float64)-np.asarray(b,np.float64)).mean(); return float(-10*np.log10(max(mse,1e-12)))
def mae_np(a,b): return float(np.abs(np.asarray(a,np.float64)-np.asarray(b,np.float64)).mean())

agg={"render_B_vs_C":{},"grad_B_vs_C":{},"psnr_vs_GT":{},"visible_vs_total":{}}
for ct in CKPTS:
    for ci in CAMS:
        key=f"{ct}_cam{ci}"
        try:
            b=np.load(f"{IMGDIR}/{ct}_B_cam{ci}.npy"); c=np.load(f"{IMGDIR}/{ct}_C_cam{ci}.npy")
            agg["render_B_vs_C"][key]={"rgb_mae":mae_np(b,c),"rgb_maxerr":float(np.abs(b-c).max()),"rgb_psnr":psnr_np(b,c)}
        except Exception as e: agg["render_B_vs_C"][key]=f"ERR:{e}"
        try:
            gb=np.load(f"{IMGDIR}/{ct}_B_cam{ci}_grad.npz"); gc=np.load(f"{IMGDIR}/{ct}_C_cam{ci}_grad.npz")
            cos={}
            for k in ["means","quats","scales","opac","colors"]:
                a=gb[k].flatten(); b_=gc[k].flatten()
                cos[k]={"cosine":float(np.dot(a,b_)/(np.linalg.norm(a)*np.linalg.norm(b_)+1e-12)),
                        "rel_l2":float(np.linalg.norm(a-b_)/(np.linalg.norm(a)+1e-12))}
            agg["grad_B_vs_C"][key]=cos
        except Exception as e: agg["grad_B_vs_C"][key]=f"ERR:{e}"
        # psnr vs GT from meta
        bm=results.get(f"{ct}_B_cam{ci}_render",{}).get("meta")
        cm=results.get(f"{ct}_C_cam{ci}_render",{}).get("meta")
        if bm and cm:
            agg["psnr_vs_GT"][key]={"B":bm["psnr_vs_gt"],"C":cm["psnr_vs_gt"],"N_total":bm["N_total"],
                                    "on_screen":bm["on_screen"],"in_frustum":bm["in_frustum"],"visible_opac":bm["visible_opac"],
                                    "opacity_mean":bm["opacity_mean"],"opacity_median":bm["opacity_median"],
                                    "num_iters":bm["num_iters"]}

out={"config":{"W":396,"H":264,"note":"identical ckpt/cam/res/bg/full-gaussian set"},"aggregate":agg}
json.dump(out,open(f"{OUT}/GP0_PHASE1_CONSISTENCY.json","w"),indent=2,default=str)
print("saved GP0_PHASE1_CONSISTENCY.json")
# summary
print("\n=== RENDER B vs C ===")
for k,v in agg["render_B_vs_C"].items(): print(f"  {k}: {v}")
print("=== GRAD B vs C (cosine) ===")
for k,v in agg["grad_B_vs_C"].items():
    if isinstance(v,dict): print(f"  {k}: "+" ".join(f"{kk}={vv['cosine']:.4f}" for kk,vv in v.items()))
    else: print(f"  {k}: {v}")
print("=== PSNR vs GT + visible ===")
for k,v in agg["psnr_vs_GT"].items(): print(f"  {k}: B={v['B']:.2f} C={v['C']:.2f} N={v['N_total']} on_screen={v['on_screen']} vis_opac={v['visible_opac']} iters={v['num_iters']}")
