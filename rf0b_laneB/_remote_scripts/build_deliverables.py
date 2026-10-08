# -*- coding: utf-8 -*-
import json, os, hashlib, io

C0 = "/root/sb0_fgs/holdout8"
FGS = "/root/rf0b_laneB/arm2_c0fgs"
OUT = "/root/rf0b_laneB/deliverables"
os.makedirs(OUT, exist_ok=True)
SCENES = ["room","garden","bicycle"]
STAGE_ITERS = [5000,15000,30000]

def load(base,s):
    p=os.path.join(base,s,"results.json")
    return json.load(open(p)) if os.path.exists(p) else None

def stage_walls(d):
    by={e["step"]:e["wall_time_s"] for e in d["eval_rows"] if e.get("tag")=="eval"}
    w5,w15,w30=by.get(5000),by.get(15000),by.get(30000)
    return {"wall_0_5k":w5,"wall_5_15k":(w15-w5) if w5 and w15 else None,
            "wall_15_30k":(w30-w15) if w15 and w30 else None,
            "total_wall_s":d["timing"]["total_wall_s"],
            "checkpoints":{"5k":w5,"15k":w15,"30k":w30}}

def gauss_integral(d):
    pi=d["per_iter"]; iters=[r["iter"] for r in pi]; ns=[r["n_gs"] for r in pi]
    by={e["step"]:e["wall_time_s"] for e in d["eval_rows"] if e.get("tag")=="eval"}
    ck=sorted(by.items())
    def wall_at(it):
        if it<=ck[0][0]: return ck[0][1]*(it/ck[0][0]) if ck[0][0] else 0.0
        for (i0,t0),(i1,t1) in zip(ck,ck[1:]):
            if i0<=it<=i1:
                f=(it-i0)/(i1-i0) if i1>i0 else 0.0
                return t0+f*(t1-t0)
        return ck[-1][1]
    ti=0.0
    for i in range(len(iters)-1):
        ti+=0.5*(ns[i]+ns[i+1])*(wall_at(iters[i+1])-wall_at(iters[i]))
    ii=0.0
    for i in range(len(iters)-1):
        ii+=0.5*(ns[i]+ns[i+1])*(iters[i+1]-iters[i])
    return {"time_integral_gauss_s":ti,"iter_integral_gauss_iters":ii,
            "final_N":ns[-1],"max_N":max(ns),
            "trajectory":[{"iter":iters[i],"n_gs":ns[i]} for i in range(0,len(iters),10)]}

def qual(d):
    fe=d["final_eval"]
    return {"psnr":fe["psnr"],"ssim":fe["ssim"],"lpips":fe["lpips"],
            "n_eval_cameras":fe["n_eval_cameras"],"wall_time_s":fe["wall_time_s"],
            "n_gaussians":fe["n_gaussians"]}

def summ(d):
    return {"scene":d["scene"],"arm":d.get("arm"),"seed":d["seed"],
            "iterations":d["iterations"],"final_N":d["final_N"],
            "initial_N":d.get("initial_N"),"peak_vram_gb":d.get("peak_vram_gb"),
            "total_clones":d.get("total_clones"),"total_splits":d.get("total_splits"),
            "total_prunes":d.get("total_prunes"),"total_opacity_resets":d.get("total_opacity_resets"),
            "timing_grade":d.get("timing_grade"),
            "quality":qual(d),"stages":stage_walls(d),"gauss":gauss_integral(d)}

raw={"c0":{},"c0fgs":{}}
for s in SCENES:
    for arm,base in [("c0",C0),("c0fgs",FGS)]:
        d=load(base,s)
        raw[arm][s]={"MISSING":os.path.join(base,s,"results.json")} if d is None else summ(d)

deltas={}
for s in SCENES:
    a=raw["c0"].get(s); b=raw["c0fgs"].get(s)
    if isinstance(a,dict) and "MISSING" not in a and isinstance(b,dict) and "MISSING" not in b:
        sp=a["stages"]["total_wall_s"]; bp=b["stages"]["total_wall_s"]
        dd={"speedup_wall":(sp/bp) if sp and bp else None,
            "c0_wall_s":sp,"c0fgs_wall_s":bp,"wall_delta_s":(bp-sp) if sp and bp else None,
            "final_N_c0":a["final_N"],"final_N_c0fgs":b["final_N"],
            "final_N_reduction_pct":(100.0*(a["final_N"]-b["final_N"])/a["final_N"]) if a["final_N"] else None,
            "int_time_c0":a["gauss"]["time_integral_gauss_s"],
            "int_time_c0fgs":b["gauss"]["time_integral_gauss_s"],
            "int_time_reduction_pct":(100.0*(a["gauss"]["time_integral_gauss_s"]-b["gauss"]["time_integral_gauss_s"])/a["gauss"]["time_integral_gauss_s"]) if a["gauss"]["time_integral_gauss_s"] else None,
            "psnr_delta":b["quality"]["psnr"]-a["quality"]["psnr"],
            "ssim_delta":b["quality"]["ssim"]-a["quality"]["ssim"],
            "lpips_delta":b["quality"]["lpips"]-a["quality"]["lpips"]}
        for st in ["wall_0_5k","wall_5_15k","wall_15_30k"]:
            av=a["stages"][st]; bv=b["stages"][st]
            dd[st+"_speedup"]=(av/bv) if av and bv else None
        deltas[s]=dd
raw["deltas"]=deltas

# ---- write individual deliverables ----
def dump(name,obj):
    io.open(os.path.join(OUT,name),"w",encoding="utf-8").write(json.dumps(obj,indent=1,ensure_ascii=False))

# RAW: full per-scene raw
dump("RF0B_RAW.json",raw)

# QUALITY: holdout metrics + deltas
q={"metric":"corrected_true_holdout (i%8==0, never trained)","seed":42,"per_scene":{}}
for s in SCENES:
    a=raw["c0"].get(s); b=raw["c0fgs"].get(s)
    if isinstance(a,dict) and "MISSING" not in a and isinstance(b,dict) and "MISSING" not in b:
        q["per_scene"][s]={"c0":a["quality"],"c0fgs":b["quality"],
            "delta":{"psnr":b["quality"]["psnr"]-a["quality"]["psnr"],
                     "ssim":b["quality"]["ssim"]-a["quality"]["ssim"],
                     "lpips":b["quality"]["lpips"]-a["quality"]["lpips"]}}
dump("RF0B_QUALITY.json",q)

# TIMING: full + stage + phase
tm={"per_scene":{}}
for s in SCENES:
    a=raw["c0"].get(s); b=raw["c0fgs"].get(s)
    if isinstance(a,dict) and "MISSING" not in a and isinstance(b,dict) and "MISSING" not in b:
        da=load(C0,s); db=load(FGS,s)
        tm["per_scene"][s]={
            "c0":{"total_wall_s":a["stages"]["total_wall_s"],"stages":a["stages"],
                 "phase_mean_ms":{k:da["timing"]["phase_ms"][k]["mean_ms"] for k in da["timing"]["phase_ms"]}},
            "c0fgs":{"total_wall_s":b["stages"]["total_wall_s"],"stages":b["stages"],
                 "phase_mean_ms":{k:db["timing"]["phase_ms"][k]["mean_ms"] for k in db["timing"]["phase_ms"]}},
            "speedup_wall":deltas[s]["speedup_wall"],
            "stage_speedup":{st:deltas[s][st+"_speedup"] for st in ["wall_0_5k","wall_5_15k","wall_15_30k"]}}
dump("RF0B_TIMING.json",tm)

# GAUSSIAN TRAJECTORY
gt={"per_scene":{}}
for s in SCENES:
    gt["per_scene"][s]={}
    for arm in ["c0","c0fgs"]:
        x=raw[arm].get(s)
        if isinstance(x,dict) and "MISSING" not in x:
            gt["per_scene"][s][arm]={"final_N":x["final_N"],"initial_N":x["initial_N"],
                "max_N":x["gauss"]["max_N"],"time_integral_gauss_s":x["gauss"]["time_integral_gauss_s"],
                "iter_integral_gauss_iters":x["gauss"]["iter_integral_gauss_iters"],
                "trajectory":x["gauss"]["trajectory"]}
    if s in deltas:
        gt["per_scene"][s]["delta"]={"final_N_reduction_pct":deltas[s]["final_N_reduction_pct"],
            "integral_reduction_pct":deltas[s]["int_time_reduction_pct"]}
dump("RF0B_GAUSSIAN_TRAJECTORY.json",gt)

# SUMMARY
dump("RF0B_SUMMARY.json",{"deltas":deltas,
    "all_scenes_complete":all(isinstance(raw["c0fgs"].get(s),dict) and "MISSING" not in raw["c0fgs"][s] for s in SCENES)})

print("BUILT deliverables (bicycle pending):", sorted(os.listdir(OUT)))
print("c0fgs bicycle present:", "MISSING" not in raw["c0fgs"].get("bicycle",{"MISSING":True}))