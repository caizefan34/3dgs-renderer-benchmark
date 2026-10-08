#!/usr/bin/env python3
"""BWD-DEPREG0-MX driver: patch a custom _C build into production diff_rasterize,
time forward/backward (CUDA events) or run a single fwd+bwd for ncu capture."""
import sys, os, json, argparse
BASE = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
for p in [BASE+"/BWD_DEPREG0_MX", BASE+"/BWD_ARITH0_MX", BASE+"/pylibs", BASE+"/BWD_DEPREG0_MX/scripts"]:
    sys.path.insert(0, p)
import torch
import mx_perf as PERF
import mx_probe as MP

class Ev:
    def __init__(s): s.e=None; s.s=None
    def __enter__(s):
        s.e=torch.cuda.Event(True); s.s=torch.cuda.Event(True); s.s.record(); return s
    def __exit__(s,*a):
        s.e.record(); torch.cuda.synchronize(); return False
    def ms(s): return s.s.elapsed_time(s.e)

def bench_fwd(fn, w, it):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(it):
        with Ev() as ev: fn()
        ts.append(ev.ms())
    ts.sort(); return ts[len(ts)//2], ts[0], ts[-1]

def bench_bwd(build_fn, w, it):
    for _ in range(w):
        img,_=build_fn(); img.sum().backward()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(it):
        img,_=build_fn(); torch.cuda.synchronize()
        with Ev() as ev: img.sum().backward()
        torch.cuda.synchronize()
        ts.append(ev.ms())
    ts.sort(); return ts[len(ts)//2], ts[0], ts[-1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--scene",required=True)
    ap.add_argument("--view",type=int,default=0)
    ap.add_argument("--warmup",type=int,default=50)
    ap.add_argument("--iters",type=int,default=100)
    ap.add_argument("--out",required=True)
    ap.add_argument("--module",default="_C_depreg0")
    ap.add_argument("--once",action="store_true")
    a=ap.parse_args()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    M=__import__(a.module)
    MP.diff_rasterize.__globals__["_C"]=M
    trainer,dataset,train_ds,sampler,ad,g=PERF.MP.build(a.config,a.checkpoint)
    views=list(train_ds); view=views[a.view]
    N=int(g.means.shape[0])
    if a.once:
        img,_=PERF.h_fwd_bwd(g,view)
        img.sum().backward()
        torch.cuda.synchronize()
        print("ONCE_DONE scene=%s N=%d"%(a.scene,N),flush=True)
        return
    h_fwd_ms,h_f0,h_f1=bench_fwd(lambda:PERF.h_fwd(g,view),a.warmup,a.iters)
    torch.cuda.empty_cache()
    h_bwd_ms,h_b0,h_b1=bench_bwd(lambda:PERF.h_fwd_bwd(g,view),a.warmup,a.iters)
    torch.cuda.empty_cache()
    res={"schema":"BWD_DEPREG0_TIMING","scene":a.scene,"view":a.view,"N":N,
         "module":a.module,"warmup":a.warmup,"iters":a.iters,
         "T_F_ms_median":round(h_fwd_ms,4),"T_F_ms_min":round(h_f0,4),
         "T_B_ms_median":round(h_bwd_ms,4),"T_B_ms_min":round(h_b0,4),
         "T_renderer_ms_median":round(h_fwd_ms+h_bwd_ms,4)}
    print("DBG_TIMING "+json.dumps(res)+" TIMING_END",flush=True)
    json.dump(res,open(a.out,"w"),indent=2)

if __name__=="__main__":
    main()
