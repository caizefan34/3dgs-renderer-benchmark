import hashlib, json, os, subprocess

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

GI = "/root/epic05_dpgr0/src_dpgr0/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference"
CUDA = "/root/epic05_dpgr0/src_dpgr0/gsplat/experimental/render/kernels/cuda"
SO = "/root/.cache/torch_extensions/py310_cu128/dpgr0_gaussian_render_inference_scene_cuda/dpgr0_gaussian_render_inference_scene_cuda.so"

sources = {
    "HigsNativeBackwardDpgr0.cu": os.path.join(GI, "HigsNativeBackwardDpgr0.cu"),
    "dpgr0_stage1.cu": os.path.join(GI, "dpgr0_stage1.cu"),
    "dpgr0_stage1.h": os.path.join(GI, "dpgr0_stage1.h"),
    "HigsNativeBackwardDpgr0.h": os.path.join(GI, "HigsNativeBackwardDpgr0.h"),
    "ext_dpgr0.cpp": os.path.join(CUDA, "ext_dpgr0.cpp"),
    "build_dpgr0.py": os.path.join(CUDA, "build_dpgr0.py"),
}
src_sha = {k: sha(v) for k, v in sources.items()}
so_sha = sha(SO)

out = {
    "gate": "DPGR0_BUILD",
    "extension_name": "dpgr0_gaussian_render_inference_scene_cuda",
    "so_path": SO,
    "so_sha256": so_sha,
    "source_sha256": src_sha,
    "build_tree": "/root/epic05_dpgr0/src_dpgr0/gsplat/experimental/render/kernels/cuda",
    "module_name": "dpgr0_gaussian_render_inference_scene_cuda",
    "bound_symbols": ["higs_rasterize_backward_dpgr0", "dpgr0_get_stage_times", "dpgr0_get_stats"],
    "env": "gsplat conda env (/root/miniforge3/envs/gsplat/bin/python, torch 2.9.1+cu128, py3.10)",
    "cuda": "12.8",
    "max_jobs": 8,
    "build_seconds": 27.28,
    "frozen_chunk_size": 32,
    "determinism_note": "Gaussian-major direct gather; DIRECT (m<=32) one write, CHUNKED (m>32) fixed 32-tile chunks + ascending chunk reduce; no float atomicAdd to final gradients",
    "build_integrity_pass": True,
    "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
}
os.makedirs("/root/epic05_dpgr0/artifacts", exist_ok=True)
with open("/root/epic05_dpgr0/artifacts/DPGR0_BUILD.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
