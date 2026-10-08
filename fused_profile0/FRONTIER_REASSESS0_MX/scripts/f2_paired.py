#!/usr/bin/env python3
"""F2 SFU paired timing: base(r0 expf) vs variant(__expf / exp2f)."""
import sys, os, json, argparse, statistics as st
BASE="/mnt/storage_pool/liaoyuanjun/fused_profile0"
for p in [BASE+"/FRONTIER_REASSESS0_MX", BASE+"/BWD_DEPREG0_MX", BASE+"/pylibs"]:
    sys.path.insert(0,p)
import torch
import mx_perf as PERF
import mx_probe as MP
import importlib.util
SO_DIR=BASE+"/FRONTIER_REASSESS0_MX"
def load_module(name):
    path=os.path.join(SO_DIR,name+".so")
    old=sys.getdlopenflags(); sys.setdlopenflags(os.RTLD_LOCAL|os.RTLD_NOW)
    try:
        spec=importlib.util.spec_from_file_location("_C_frontier0",path)
        mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod
        spec.loader.exec_module(mod)
    finally:
        sys.setdlopenflags(old)
    return mod
class Ev:
    def __init__(s): s.e=None; s.s=None
    def __enter__(s): s.e=torch.cuda.Event(True); s.s=torch.cuda.Event(True); s.s.record(); return s
    def __exit__(s,*a): s.e.record(); torch.cuda.synchronize(); return False
    def ms(s): return s.s.elapsed_time(s.e)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True); ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--scene",required=True); ap.add_argument("--view",type=int,default=0)
    ap.add_argument("--base",default="_C_frontier0_r0"); ap.add_argument("--variant",required=True)
    ap.add_argument("--warmup",type=int,default=30); ap.add_argument("--iters",type=int,default=120)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    MB=load_module(a.base); MV=load_module(a.variant)
    trainer,dataset,train_ds,sampler,ad,g=PERF.MP.build(a.config,a.checkpoint)
    views=list(train_ds); view=views[a.view]
    def bwd_with(mod):
        MP.diff_rasterize.__globals__["_C"]=mod
        img,_=PERF.h_fwd_bwd(g,view); return img
    for _ in range(a.warmup):
        img=bwd_with(MB); img.sum().backward()
        img=bwd_with(MV); img.sum().backward()
    torch.cuda.synchronize()
    pairs=[]
    for _ in range(a.iters):
        img=bwd_with(MB); torch.cuda.synchronize()
        with Ev() as ev: img.sum().backward()
        torch.cuda.synchronize(); tb=ev.ms()
        img=bwd_with(MV); torch.cuda.synchronize()
        with Ev() as ev: img.sum().backward()
        torch.cuda.synchronize(); tv=ev.ms()
        pairs.append((tb,tv))
    tb=[p[0] for p in pairs]; tv=[p[1] for p in pairs]
    out={"schema":"FRONTIER_REASSESS0_PAIRED","scene":a.scene,"variant":a.variant,
         "N":int(g.means.shape[0]),"base":a.base,"warmup":a.warmup,"iters":a.iters,
         "base_bwd_median":st.median(tb),"base_bwd_min":min(tb),
         "variant_bwd_median":st.median(tv),"variant_bwd_min":min(tv),
         "S_backward_median":st.median(tb)/st.median(tv),
         "S_backward_min":min(tb)/min(tv),
         "S_backward_paired_median":st.median([x/y for x,y in pairs]),
         "delta_T_bwd_ms_median":st.median(tv)-st.median(tb)}
    json.dump(out,open(a.out,"w"),indent=2)
    print("WROTE",a.out,"S_med=%.4f S_min=%.4f S_paired=%.4f"%(out["S_backward_median"],out["S_backward_min"],out["S_backward_paired_median"]))
main()
