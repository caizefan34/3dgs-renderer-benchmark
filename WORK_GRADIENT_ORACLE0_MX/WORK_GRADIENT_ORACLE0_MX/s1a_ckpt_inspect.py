#!/usr/bin/env python3
"""Inspect checkpoint for PCA keys and confirm color convention."""
import torch, json
CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
b = torch.load(CKPT, map_location="cpu", weights_only=False)
print("=== top-level keys ===")
print(list(b.keys()))
print("model_name:", b.get("model_name"))
print("num_iterations_trained:", b.get("num_iterations_trained"))
print("SH_DEGREE:", b.get("SH_DEGREE"))
sd = b.get("model_state_dict", {})
non_g = [k for k in sd.keys() if not k.startswith("gaussians.")]
print("=== non-gaussian state_dict keys ===")
print(non_g)
# Search entire checkpoint (including any nested config) for pca
def scan(obj, path="root", hits=None):
    if hits is None: hits = []
    if isinstance(obj, dict):
        for k,v in obj.items():
            if "pca" in str(k).lower():
                hits.append((path+"."+str(k), type(v).__name__))
            scan(v, path+"."+str(k), hits)
    elif isinstance(obj, (list,tuple)):
        for i,v in enumerate(obj[:5]):
            scan(v, f"{path}[{i}]", hits)
    return hits
hits = scan(b)
print("=== PCA references anywhere in checkpoint ===")
print(hits if hits else "NONE")
# Also dump any config-ish keys
for k in b.keys():
    if k not in ("model_state_dict",):
        v = b[k]
        if isinstance(v, (dict,list)):
            print(f"--- {k} (type {type(v).__name__}) ---")
            print(json.dumps(v, default=str)[:800] if not isinstance(v,dict) else list(v.keys())[:20])
