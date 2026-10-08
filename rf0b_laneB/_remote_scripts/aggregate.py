import json, os, math

C0 = "/root/sb0_fgs/holdout8"
FGS = "/root/rf0b_laneB/arm2_c0fgs"
SCENES = ["room","garden","bicycle"]

def load(base, s):
    p = os.path.join(base, s, "results.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))

def stage_walls(d):
    by = {e["step"]: e["wall_time_s"] for e in d["eval_rows"] if e.get("tag")=="eval"}
    w5  = by.get(5000); w15 = by.get(15000); w30 = by.get(30000)
    return {
        "wall_0_5k":  w5,
        "wall_5_15k": (w15-w5) if (w5 is not None and w15 is not None) else None,
        "wall_15_30k":(w30-w15) if (w15 is not None and w30 is not None) else None,
        "total_wall_s": d["timing"]["total_wall_s"],
        "ckpt": {"5k":w5,"15k":w15,"30k":w30},
    }

def gaussian_integral(d):
    pi = d["per_iter"]
    iters = [r["iter"] for r in pi]
    ns    = [r["n_gs"] for r in pi]
    by = {e["step"]: e["wall_time_s"] for e in d["eval_rows"] if e.get("tag")=="eval"}
    ck = sorted(by.items())
    def wall_at(it):
        if it <= ck[0][0]:
            return ck[0][1]*(it/ck[0][0]) if ck[0][0] else 0.0
        for (i0,t0),(i1,t1) in zip(ck, ck[1:]):
            if i0 <= it <= i1:
                f = (it-i0)/(i1-i0) if i1>i0 else 0.0
                return t0 + f*(t1-t0)
        return ck[-1][1]
    integral = 0.0
    for i in range(len(iters)-1):
        i0,i1 = iters[i], iters[i+1]
        dt = wall_at(i1)-wall_at(i0)
        integral += 0.5*(ns[i]+ns[i+1])*dt
    int_iter = 0.0
    for i in range(len(iters)-1):
        di = iters[i+1]-iters[i]
        int_iter += 0.5*(ns[i]+ns[i+1])*di
    return {"time_integral_gauss_s": integral, "iter_integral_gauss_iters": int_iter,
            "final_N": ns[-1], "max_N": max(ns)}

def q(d):
    fe = d["final_eval"]
    return {"psnr": fe["psnr"], "ssim": fe["ssim"], "lpips": fe["lpips"],
            "n_eval_cameras": fe["n_eval_cameras"], "n_gaussians": fe["n_gaussians"]}

def summarize(d):
    return {
        "scene": d["scene"], "arm": d.get("arm"), "seed": d["seed"],
        "iterations": d["iterations"], "final_N": d["final_N"],
        "initial_N": d.get("initial_N"), "peak_vram_gb": d.get("peak_vram_gb"),
        "total_clones": d.get("total_clones"), "total_splits": d.get("total_splits"),
        "total_prunes": d.get("total_prunes"), "total_opacity_resets": d.get("total_opacity_resets"),
        "timing_grade": d.get("timing_grade"),
        "quality": q(d), "stages": stage_walls(d), "gauss": gaussian_integral(d),
    }

out = {"c0":{}, "c0fgs":{}}
for s in SCENES:
    for arm,base in [("c0",C0),("c0fgs",FGS)]:
        d = load(base,s)
        out[arm][s] = {"MISSING": os.path.join(base,s,"results.json")} if d is None else summarize(d)

out["deltas"] = {}
for s in SCENES:
    a = out["c0"].get(s); b = out["c0fgs"].get(s)
    if isinstance(a,dict) and "MISSING" not in a and isinstance(b,dict) and "MISSING" not in b:
        sp = a["stages"]["total_wall_s"]; bp = b["stages"]["total_wall_s"]
        dd = {"speedup_wall": (sp/bp) if (sp and bp) else None,
              "c0_wall_s": sp, "c0fgs_wall_s": bp,
              "wall_delta_s": (bp-sp) if (sp and bp) else None,
              "final_N_c0": a["final_N"], "final_N_c0fgs": b["final_N"],
              "final_N_reduction_pct": (100.0*(a["final_N"]-b["final_N"])/a["final_N"]) if a["final_N"] else None,
              "int_time_c0": a["gauss"]["time_integral_gauss_s"],
              "int_time_c0fgs": b["gauss"]["time_integral_gauss_s"],
              "int_time_reduction_pct": (100.0*(a["gauss"]["time_integral_gauss_s"]-b["gauss"]["time_integral_gauss_s"])/a["gauss"]["time_integral_gauss_s"]) if a["gauss"]["time_integral_gauss_s"] else None,
              "psnr_delta": b["quality"]["psnr"]-a["quality"]["psnr"],
              "ssim_delta": b["quality"]["ssim"]-a["quality"]["ssim"],
              "lpips_delta": b["quality"]["lpips"]-a["quality"]["lpips"]}
        for st in ["wall_0_5k","wall_5_15k","wall_15_30k"]:
            av=a["stages"][st]; bv=b["stages"][st]
            dd[st+"_speedup"] = (av/bv) if (av and bv) else None
        out["deltas"][s] = dd
print(json.dumps(out, indent=1))