import json
import os
from pathlib import Path

# 13-scene standard benchmark suite
SCENES = [
    "bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
    "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"
]

# Known F0 checkpoints (authoritative, hash-verified)
F0_CHECKPOINTS = {
    "room": "/mnt/storage_pool/liaoyuanjun/cgt_f0_fused_cuda/results/B1_room_multiphase_v2/checkpoints/iter_15000.pt",
    "garden": "/mnt/storage_pool/liaoyuanjun/cgt_f0_fused_cuda/results/B1_garden_30k/checkpoints/iter_30000.pt",
    "bicycle": "/mnt/storage_pool/liaoyuanjun/cgt_f0_fused_cuda/results/B1_bicycle_30k/checkpoints/iter_30000.pt"
}

# Data paths (from handoff summary)
DATASET_ROOT = Path("/tmp/3dgs-renderer-benchmark/data/datasets")

def get_data_path(scene):
    """Infer data path for each scene."""
    # drjohnson, playroom, train, truck are directly under datasets/
    if scene in ["drjohnson", "playroom", "train", "truck"]:
        return DATASET_ROOT / scene
    # Others under mipnerf360/
    return DATASET_ROOT / "mipnerf360" / scene

def audit_scene(scene):
    """Audit one scene: data path, checkpoint, metadata."""
    data_path = get_data_path(scene)
    ckpt_path = F0_CHECKPOINTS.get(scene, None)
    
    result = {
        "scene": scene,
        "data_path": str(data_path),
        "data_exists": data_path.exists(),
        "checkpoint_path": ckpt_path,
        "checkpoint_exists": ckpt_path.exists() if ckpt_path else False,
        "F0_checkpoint": scene in F0_CHECKPOINTS
    }
    
    # If data exists, try to infer camera count, resolution
    if result["data_exists"]:
        images_dir = data_path / "images"
        sparse_dir = data_path / "sparse" / "0"
        if images_dir.exists():
            result["images_count"] = len(list(images_dir.glob("*")))
        if sparse_dir.exists():
            result["sparse_exists"] = True
            cameras_file = sparse_dir / "cameras.bin"
            result["cameras_file_exists"] = cameras_file.exists()
    
    # Classify
    if result["F0_checkpoint"] and result["checkpoint_exists"] and result["data_exists"]:
        result["classification"] = "EXACT_COHORT"
    elif result["data_exists"] and not result["F0_checkpoint"]:
        result["classification"] = "MISSING_CHECKPOINT"
    elif not result["data_exists"]:
        result["classification"] = "MISSING_DATA"
    else:
        result["classification"] = "UNKNOWN"
    
    return result

def main():
    registry = []
    for scene in SCENES:
        info = audit_scene(scene)
        registry.append(info)
        print(f"{scene:12s} -> {info['classification']:20s} data={info['data_exists']} ckpt={info['checkpoint_exists']}")
    
    output = {
        "registry": registry,
        "summary": {
            "total_scenes": len(SCENES),
            "exact_cohort": sum(1 for r in registry if r["classification"] == "EXACT_COHORT"),
            "missing_checkpoint": sum(1 for r in registry if r["classification"] == "MISSING_CHECKPOINT"),
            "missing_data": sum(1 for r in registry if r["classification"] == "MISSING_DATA"),
        }
    }
    
    out_path = Path(__file__).parent.parent / "results" / "PAPER13_SCENE_REGISTRY.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
