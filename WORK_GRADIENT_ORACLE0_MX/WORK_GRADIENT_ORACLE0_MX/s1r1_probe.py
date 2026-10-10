#!/usr/bin/env python3
import torch, json, os
CK = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
for f in ["0005000.pt","0010000.pt","final.pt"]:
    p=f"{CK}/{f}"
    b=torch.load(p,map_location="cpu",weights_only=False)
    print(f, "num_iters=", b.get("num_iterations_trained"), "model=", b.get("model_name"),
          "N=", b["model_state_dict"]["gaussians._means"].shape[0] if "gaussians._means" in b.get("model_state_dict",{}) else "?")
cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
print("eval_cameras count:", len(cams))
print("cam0:", {k:cams[0][k] for k in cams[0] if k!='rotation'})
print("rotation cam0 (first 3 rows):", cams[0]["rotation"][:3])
base="/mnt/storage_pool/liaoyuanjun/fused_profile0/faster-gaussian-splatting"
print("FasterGS dir exists:", os.path.isdir(base))
print("entries:", os.listdir(base)[:20])
