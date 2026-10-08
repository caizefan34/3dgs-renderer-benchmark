import json, hashlib, os, datetime
ROOT="amps_gs/AMPS0"
def sha(p):
    if not os.path.exists(p): return None
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(65536),b""): h.update(b)
    return h.hexdigest()

fgs_env=json.load(open("fgs_strong_baseline_final/FGS_ENVIRONMENT.json",encoding="utf-8"))
fgs_lock=json.load(open("fgs_strong_baseline_final/FGS_SOURCE_LOCK.json",encoding="utf-8"))
fgs_tim=json.load(open("fgs_strong_baseline_final/FGS_TIMING.json",encoding="utf-8"))
fgs_cfg=json.load(open("fgs_strong_baseline_final/FGS_CONFIGS.json",encoding="utf-8"))
fgs_qual=json.load(open("fgs_strong_baseline_final/FGS_QUALITY.json",encoding="utf-8"))

runs=fgs_tim["runs"]
def runrow(name):
    r=runs[name]
    return {"wall_s":r["wall_s_external"],"framework_total_time_s":r["framework_total_time_s"],
            "training_iteration_s":r["training_iteration_s"],"peak_vram_gib":round(r["peak_vram_allocated_bytes"]/1024**3,3),
            "n_gaussians_final":r["n_gaussians_final"],"exit_code":r.get("exit_code")}

baseline={
 "schema":"AMPS0_BASELINE_AUDIT","generated":datetime.date.today().isoformat(),
 "role":"strong_baseline","primary_target":"Faster-GS fused training backend",
 "reason":"Objective section 2: primary target is official/reproducible Faster-GS fused training backend. Faster-GS fused beats frozen C0 (gsplat) by 1.45-1.76x cross-backend on all 3 scenes (room/garden/bicycle). Vanilla 3DGS and clean gsplat retained only as diagnostic secondary baselines, never for headline speed.",
 "frozen":{
   "repo":"nerficg-project/faster-gaussian-splatting (NeRFICG framework)",
   "framework_commit":fgs_lock["nerficg_framework"]["commit"],
   "method_lane":"FasterGSFused",
   "method_commit":fgs_lock["methods"]["FasterGSFused"]["commit"],
   "remote":"https://github.com/nerficg-project/faster-gaussian-splatting",
   "cuda_backend":"FasterGSFusedCudaBackend (pip-installed, non-editable)",
   "cuda_backend_so_sha256":fgs_lock["cuda_extension_so_sha256"].get("/root/miniforge3/envs/p16_fastergs/lib/python3.11/site-packages/FasterGSFusedCudaBackend/_C.cpython-311-x86_64-linux-gnu.so"),
   "toolchain":{"gpu":fgs_env["gpu"],"python":fgs_env["python"],"torch":fgs_env["torch"],
     "cuda_runtime":fgs_env["cuda_runtime"],"driver":fgs_env["driver"],
     "compiler":fgs_env["compiler"],"cudnn":fgs_env["cudnn"],"conda_env":fgs_env["conda_env"]},
   "authoritative_timing_host":"EPIC-05 A100-SXM4-80GB (GPU index 1)","local_gpu_used":False,
   "timing_discipline":fgs_tim["timing_discipline"],
   "evidence_dir":"fgs_strong_baseline_final/",
   "evidence_sha256":{
     "FGS_TIMING.json":sha("fgs_strong_baseline_final/FGS_TIMING.json"),
     "FGS_CONFIGS.json":sha("fgs_strong_baseline_final/FGS_CONFIGS.json"),
     "FGS_SOURCE_LOCK.json":sha("fgs_strong_baseline_final/FGS_SOURCE_LOCK.json"),
     "FGS_ENVIRONMENT.json":sha("fgs_strong_baseline_final/FGS_ENVIRONMENT.json"),
     "FGS_QUALITY.json":sha("fgs_strong_baseline_final/FGS_QUALITY.json")}
 },
 "fused_runs":{n:runrow(n) for n in runs if "fused" in n},
 "per_scene_config_room":{k:v for k,v in fgs_cfg.items() if k.endswith("_iso_room_s42_30000")},
 "quality":fgs_qual.get("quality",fgs_qual),
 "local_reproducibility":{
   "local_gpu":"NVIDIA GeForce RTX 5070 Laptop (8GB)","local_torch":"2.13.0+cu130","local_gsplat":"1.4.0",
   "faster_gs_fused_buildable_locally":False,
   "reason":"Faster-GS fused CUDA backend needs A100 toolchain (CUDA 12.8, gcc 11.4, NeRFICG framework) absent locally; gsplat 1.4 CUDA cannot JIT-compile here (no MSVC). Local work limited to pure-PyTorch oracle numerics, marked non-authoritative.",
   "shared_gpu_numerics_only":True,"performance_evidence_valid_locally":False
 },
 "diagnostic_secondary_baselines":{"vanilla_3dgs":"diagnostic only","clean_gsplat":"C0 frozen backend, diagnostic only"}
}
json.dump(baseline, open(ROOT+"/AMPS0_BASELINE_AUDIT.json","w"), indent=2)
print("BASELINE_AUDIT written; fused room wall_s:", baseline["fused_runs"]["fgs_fused_iso_room_s42_30000"]["wall_s"])
print("fused so sha:", baseline["frozen"]["cuda_backend_so_sha256"][:16])
