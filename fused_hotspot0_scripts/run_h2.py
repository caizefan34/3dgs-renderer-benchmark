#!/usr/bin/env python3
"""FUSED-HOTSPOT0 H2 census driver: per-kernel GPU self-time via chrome trace."""
import subprocess, os, sys, json, time, glob, collections

P = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
PY = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3.10"
HOT = P + "/FUSED_HOTSPOT0"
RAW = HOT + "/raw"
PROF = HOT + "/profiler"
os.makedirs(RAW, exist_ok=True); os.makedirs(PROF, exist_ok=True)

STAGE_CKPT = {"s1_5k":"0005000.pt","s2_10k":"0010000.pt","s3_mature":"final.pt"}
CKPT_ROOT = {
    "room":   P+"/../strong_baselines/nerficg/output/FasterGSFused/fp1_room_2026-10-06-01-01-33/checkpoints",
    "bicycle":P+"/../strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints",
    "garden": P+"/../strong_baselines/nerficg/output/FasterGSFused/fp1_garden_2026-10-06-01-32-48/checkpoints",
}
CKPT_ROOT = {k: os.path.normpath(v) for k,v in CKPT_ROOT.items()}

ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"]="0"; ENV["TMPDIR"]="/dev/shm"; ENV["TEMP"]="/dev/shm"; ENV["TMP"]="/dev/shm"
ENV["PYTHONPATH"] = ":".join([P+"/pylibs", P+"/methods_overlay",
    "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/src",
    "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/scripts"])
os.environ['CUDA_VISIBLE_DEVICES']='0'; os.environ['TMPDIR']='/dev/shm'; os.environ['TEMP']='/dev/shm'; os.environ['TMP']='/dev/shm'
for _p in [P+'/pylibs', P+'/methods_overlay', '/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/src', '/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/scripts']:
    if _p not in sys.path: sys.path.insert(0, _p)

def nvidia():
    return subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()

def guard():
    t0=time.time(); s=[]
    while time.time()-t0 < 21:
        ln=nvidia()[0].split(","); s.append((int(ln[1]),int(ln[2]))); time.sleep(2)
    mxm=max(m for m,_ in s); mxu=max(u for _,u in s)
    ok=(mxm<=512) and (mxu<=2)
    rec={"schema":"FUSED_HOTSPOT0_GUARD","phase":"H2","gpu":"0","observation_sec":round(time.time()-t0,1),
         "n_samples":len(s),"max_mem_observed_mib":mxm,"max_util_observed_pct":mxu,"clean":ok,"raw_samples":s}
    json.dump(rec, open(RAW+"/guard_h2_gpu0.json","w"), indent=2); return ok,rec

def parse_trace(path):
    """Aggregate chrome-trace kernel + api + memcpy events by name."""
    import json as _json
    ev = _json.load(open(path))["traceEvents"]
    kern = collections.defaultdict(lambda:{"count":0,"total_us":0,"durs":[]})
    other = collections.defaultdict(lambda:{"count":0,"total_us":0})
    cats = collections.Counter()
    for e in ev:
        cat = e.get("cat",""); cats[cat]+=1
        dur = e.get("dur",0) or 0
        name = e.get("name","?")
        if cat=="kernel":
            kern[name]["count"]+=1; kern[name]["total_us"]+=dur; kern[name]["durs"].append(dur)
        elif cat in ("cuda_runtime","cuda_driver"):
            other["api:"+name]["count"]+=1; other["api:"+name]["total_us"]+=dur
        elif "Memcpy" in name or "Memset" in name or cat in ("Memcpy","Memset"):
            other[name]["count"]+=1; other[name]["total_us"]+=dur
    return kern, other, cats

def run(scene, stage):
    sys.path.insert(0, HOT+"/scripts")
    import importlib, hotspot
    importlib.reload(hotspot)
    ck = os.path.join(CKPT_ROOT[scene], STAGE_CKPT[stage])
    tag=f"{scene}_{stage}"
    out_trace = f"{PROF}/census_{tag}_trace.json"
    trainer, dataset, train_ds, sampler, ad, g = hotspot.build(f"{P}/profile/mat_{scene}.yaml", ck, 2)
    _iter = hotspot.make_iter(trainer.renderer, trainer.loss)
    warmup=30; N=30
    for i in range(warmup): _iter(train_ds, sampler, ad, g, 100000+i)
    torch_ok=True
    import torch
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(N): _iter(train_ds, sampler, ad, g, 200000+i)
        torch.cuda.synchronize()
    prof.export_chrome_trace(out_trace)
    kern, other, cats = parse_trace(out_trace)
    # clean H1 total for normalization
    h1 = json.load(open(f"{RAW}/clean_{scene}_{stage}.json"))
    T_iter = h1["median_total_ms"]*1000.0
    # per-iter kernel aggregation
    krows=[]
    tot_kern_us=0
    for name,d in kern.items():
        tot_kern_us += d["total_us"]/N
        durs=sorted(d["durs"])
        med = durs[len(durs)//2] if durs else 0
        krows.append({"name":name,"count_per_iter":round(d["count"]/N,3),
                      "total_us_per_iter":round(d["total_us"]/N,2),
                      "mean_us":round(d["total_us"]/max(1,d["count"]),2),
                      "median_us":round(med,2)})
    krows.sort(key=lambda r:-r["total_us_per_iter"])
    for r in krows:
        r["pct_kernel_time"]=round(100*r["total_us_per_iter"]/max(1e-9,tot_kern_us),2)
        r["pct_iteration_time"]=round(100*r["total_us_per_iter"]/max(1e-9,T_iter),2)
    orows=[{"name":k,"count":v["count"],"total_us":round(v["total_us"],1)} for k,v in other.items()]
    orows.sort(key=lambda r:-r["total_us"])
    out={"schema":"FUSED_HOTSPOT0_KERNEL_CENSUS","scene":scene,"stage":stage,"mode":"B_census_chrome_trace",
         "preloading_level":2,"iters_profiled":N,"warmup":warmup,"trace":out_trace,
         "T_iter_clean_ms":round(T_iter/1000.0,4),
         "total_kernel_us_per_iter":round(tot_kern_us,2),
         "total_kernel_ms_per_iter":round(tot_kern_us/1000.0,4),
         "kernels":krows,"api_and_memcpy":orows,"trace_categories":dict(cats)}
    json.dump(out, open(f"{RAW}/census_{scene}_{stage}.json","w"), indent=2)
    print(f"CENDONE {tag} kernels={len(krows)} tot_kern_ms={tot_kern_us/1000:.3f} T_iter={T_iter/1000:.3f}", flush=True)
    return out

if __name__=="__main__":
    print("=== H2 GUARD ===", flush=True)
    ok,rec=guard(); print("GUARD", json.dumps({k:rec[k] for k in ["clean","max_mem_observed_mib","max_util_observed_pct"]}), flush=True)
    if not ok: print("GUARD_FAIL", flush=True); sys.exit(2)
    for scene in ["room","bicycle","garden"]:
        for stage in ["s1_5k","s2_10k","s3_mature"]:
            run(scene, stage)
    print("H2_ALL_DONE", flush=True)