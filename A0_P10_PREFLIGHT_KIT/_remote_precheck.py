import sys, os, hashlib, json
PY = "/root/miniforge3/envs/gsplat/bin/python"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
print("=== EPIC-05 precheck ===")
try:
    import torch
    print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
          "n_gpu", torch.cuda.device_count(), "gpu0", torch.cuda.get_device_name(0))
except Exception as e:
    print("TORCH_IMPORT_FAIL", e)
print("--- worktree kernel sources ---")
for p in ["/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/cuda/csrc/HigsNativeBackward.cu",
          "/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh"]:
    if os.path.exists(p):
        print("EXISTS", sha(p)[:24], p)
    else:
        print("MISSING", p)
print("--- room cameras.json cam19 ---")
cj = "/root/3dgs-renderer-benchmark/data/official/mipnerf360/room/cameras.json"
if os.path.exists(cj):
    c = json.load(open(cj))
    print("Ncams", len(c), "cam19_img", c[19].get("img_name"))
else:
    print("MISSING cameras.json", cj)
print("--- key frozen artifacts present? ---")
for p in ["/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so",
          "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so",
          "/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_25000.pt"]:
    print("OK " if os.path.exists(p) else "MISSING", p)
print("=== precheck done ===")
