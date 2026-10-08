#!/usr/bin/env python3
"""Load each scene ONCE, time multiple _C builds (forward+backward, CUDA events).
Paired same-session ratios cancel cross-session GPU contention. Usage:
  depreg_multitim.py --config C --checkpoint K --scene S --view 0 \
     --modules _C_depreg0,_C_depreg0_dep --warmup 50 --iters 100 --out out.json"""
import sys, os, json, argparse
BASE="/mnt/storage_pool/liaoyuanjun/fused_profile0"
for p in [BASE+"/BWD_DEPREG0_MX", BASE+"/BWD_ARITH0_MX", BASE+"/pylibs"]:
    sys.path.insert(0,p)
import torch
import mx_perf as PERF
import mx_probe as MP
import importlib.util, sys, os
SO_DIR="/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_DEPREG0_MX"
def load_module(name):
    """Load _C_depreg0_<tag>.so (built with PYBIND11_MODULE(_C_depreg0)).
    RTLD_LOCAL avoids PyInit__C_depreg0 symbol collision across .so files."""
    path=os.path.join(SO_DIR, name+".so")
    old=sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_LOCAL | os.RTLD_NOW)
    try:
        spec=importlib.util.spec_from_file_location("_C_depreg0", path)
        mod=importlib.util.module_from_spec(spec)
        sys.modules[name]=mod
        spec.loader.exec_module(mod)
    finally:
        sys.setdlopenflags(old)
    return mod

class Ev:
    def __init__(s): s.e=None; s.s=None
    def __enter__(s):
        s.e=torch.cuda.Event(True); s.s=torch.cuda.Event(True); s.s.record(); return s
    def __exit__(s,*a): s.e.record(); torch.cuda.synchronize(); return False
    def ms(s): return s.s.elapsed_time(s.e)

def bench(bld_fn,w,it):
    for _ in range(w):
        img,_=bld_fn(); img.sum().backward()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(it):
        img,_=bld_fn(); torch.cuda.synchronize()
        with Ev() as ev: img.sum().backward()
        torch.cuda.synchronize()
        ts.append(ev.ms())
    ts.sort()
    return {"median":round(ts[len(ts)//2],4),"min":round(ts[0],4),"max":round(ts[-1],4),
            "n":len(ts),"all":ts}

def bench_fwd(fn,w,it):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(it):
        with Ev() as ev: fn()
        ts.append(ev.ms())
    ts.sort()
    return {"median":round(ts[len(ts)//2],4),"min":round(ts[0],4)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--scene",required=True)
    ap.add_argument("--view",type=int,default=0)
    ap.add_argument("--warmup",type=int,default=50)
    ap.add_argument("--iters",type=int,default=100)
    ap.add_argument("--modules",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    mods=[m for m in a.modules.split(",") if m]
    trainer,dataset,train_ds,sampler,ad,g=PERF.MP.build(a.config,a.checkpoint)
    views=list(train_ds); view=views[a.view]
    N=int(g.means.shape[0])
    # warm the production path once
    _=PERF.h_fwd(g,view); torch.cuda.synchronize()
    res={"schema":"BWD_DEPREG0_MULTITIM","scene":a.scene,"view":a.view,"N":N,
         "warmup":a.warmup,"iters":a.iters,"modules":{}}
    for m in mods:
        M=load_module(m)
        MP.diff_rasterize.__globals__["_C"]=M
        fwd=bench_fwd(lambda:PERF.h_fwd(g,view),a.warmup,a.iters)
        torch.cuda.empty_cache()
        bwd=bench(lambda:PERF.h_fwd_bwd(g,view),a.warmup,a.iters)
        torch.cuda.empty_cache()
        res["modules"][m]={"forward":fwd,"backward":bwd,
            "T_renderer_ms":round(fwd["median"]+bwd["median"],4)}
        print("MOD %s fwd_med=%s bwd_med=%s bwd_min=%s"%(m,fwd["median"],bwd["median"],bwd["min"]),flush=True)
    # ratios vs first module
    base=mods[0]
    res["ratios_vs_"+base]={}
    for m in mods:
        if m==base: continue
        rb=res["modules"][base]["backward"]["median"]
        rm=res["modules"][m]["backward"]["median"]
        rr0=res["modules"][base]["T_renderer_ms"]
        rrm=res["modules"][m]["T_renderer_ms"]
        res["ratios_vs_"+base][m]={
            "S_backward":round(rb/rm,4) if rm else None,
            "S_renderer":round(rr0/rrm,4) if rrm else None,
            "delta_T_bwd_ms":round(rb-rm,4),
            "delta_T_renderer_ms":round(rr0-rrm,4)}
    print("RATIOS "+json.dumps(res["ratios_vs_"+base]),flush=True)
    json.dump(res,open(a.out,"w"),indent=2)

if __name__=="__main__":
    main()
