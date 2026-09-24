"""Inspect camera format + dataset structure for H1 scenes on mx."""
import json, os, glob

SCENES = {
    "train": {
        "result_dir": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native",
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/point_cloud/iteration_30000/point_cloud.ply",
        "data_dir": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official",
    },
    "room": {
        "result_dir": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native",
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
        "data_dir": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room",
    },
    "bicycle": {
        "result_dir": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native",
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
        "data_dir": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle",
    },
}

for name, info in SCENES.items():
    print(f"\n=== {name} ===")
    print("ply_exists", os.path.exists(info["ply"]), info["ply"])
    rd = info["result_dir"]
    print("result_dir_contents", sorted(os.listdir(rd)) if os.path.isdir(rd) else "MISSING")
    # check cameras.json
    cams_path = os.path.join(rd, "cameras.json")
    if os.path.exists(cams_path):
        with open(cams_path) as f:
            cams = json.load(f)
        print("n_cams", len(cams))
        if cams:
            c0 = cams[0]
            print("cam0_keys", sorted(c0.keys()))
            for k, v in c0.items():
                if isinstance(v, list):
                    print(f"  {k}: list[{len(v)}] first={v[:3]}")
                else:
                    print(f"  {k}: {v}")
    else:
        print("no cameras.json at", cams_path)
    # check data dir for images
    dd = info["data_dir"]
    if os.path.isdir(dd):
        print("data_dir_contents", sorted(os.listdir(dd))[:20])
        # find images
        imgs = []
        for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG"):
            imgs.extend(glob.glob(os.path.join(dd, "**", ext), recursive=True))
        print("n_images", len(imgs), "first3", [os.path.basename(x) for x in imgs[:3]])
    else:
        print("data_dir MISSING", dd)

# Also check the repo data dir for train scene
print("\n=== SEARCH TRAIN DATA ===")
for root in ["/mnt/storage_pool/3dgs-renderer-benchmark/repo/data",
             "/mnt/storage_pool/3dgs-renderer-benchmark/repo/datasets"]:
    if os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            full = os.path.join(root, d)
            if "tanks" in d.lower() or "train" in d.lower():
                print("found", full)
                if os.path.isdir(full):
                    print("  contents", sorted(os.listdir(full))[:15])

# Check the repo data/scenes which might have the colmap data
print("\n=== REPO data/scenes ===")
scenes_dir = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/scenes"
if os.path.isdir(scenes_dir):
    print("contents", sorted(os.listdir(scenes_dir))[:30])

print("\n=== INSPECT_DONE ===")
