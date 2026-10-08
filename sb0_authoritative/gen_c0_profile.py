import json, os, statistics as st
BASE="/root/3dgs-publication-epic05/full30k"
SCENES=["room","bicycle","garden"]
SEEDS=["s42","s43","s44"]
def load(scene,seed):
    p=f"{BASE}/c0_{seed}_{scene}"
    r=json.load(open(f"{p}/results.json"))
    t=json.load(open(f"{p}/timing.json"))
    fe=r["final_eval"]
    ph=t.get("phase_ms",{})
    return {
        "psnr":fe["psnr"],"ssim":fe["ssim"],"lpips":fe["lpips"],
        "n_gaussians":fe["n_gaussians"],"wall_s":fe["wall_time_s"],"n_eval_cameras":fe["n_eval_cameras"],
        "mean_iter_ms":t["mean_iter_ms"],"total_wall_s":t["total_wall_s"],"iters_per_s":t["iterations_per_s"],
        "forward_ms":t["forward_ms"],"loss_ms":t["loss_ms"],"backward_ms":t["backward_ms"],
        "optimizer_ms":t["optimizer_ms"],
        "phase_ms":{k:{kk:vv for kk,vv in v.items()} for k,v in ph.items()},
        "peak_vram_gb":r.get("peak_vram_gb"),
    }
out={"task":"SB0-FGS","deliverable":"SB0_C0_PROFILE","source":"EPIC-05 A100-SXM4-80GB (authoritative)",
     "env":"torch 2.9.1+cu128 (C0 runs); timing_grade=PUBLICATION","c0_binary_identity_match":True,
     "note":"C0 = F9+SCALAR_ADJOINT+H8_MR, PX=2, eps2d=0.3, HIGS_BWD_ABSGRAD=1, densify 500..15000, tile16, sh3, lambda_dssim=0.2, resolution=1080p, seed=42/43/44, 30K.",
     "scenes":{}}
for sc in SCENES:
    runs={sd:load(sc,sd) for sd in SEEDS if os.path.isdir(f"{BASE}/c0_{sd}_{sc}")}
    agg={}
    for metric in ["psnr","ssim","lpips","n_gaussians","mean_iter_ms","forward_ms","loss_ms","backward_ms","optimizer_ms","wall_s","peak_vram_gb"]:
        vals=[runs[sd][metric] for sd in runs]
        agg[metric]={"mean":round(st.mean(vals),4),"min":round(min(vals),4),"max":round(max(vals),4),"std":round(st.pstdev(vals),4) if len(vals)>1 else 0.0}
    out["scenes"][sc]={"runs":runs,"aggregate":agg}
print(json.dumps(out,indent=2))
