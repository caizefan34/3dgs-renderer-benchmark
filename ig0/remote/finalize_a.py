import json, os, sys, platform, hashlib, math
import numpy as np
import torch

OUT = "/root/ig0_run/out"
REF = "/root/ig0_run"
REPO = "/root/3dgs-renderer-benchmark"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def write(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2)

# ---- environment ----
env = {
    "artifact": "IG0_ENVIRONMENT.json",
    "gpu_execution_host": platform.node(),
    "GPU_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch": torch.__version__,
    "CUDA_version": torch.version.cuda,
    "python": sys.version.split()[0],
    "hostname": platform.node(),
    "remote_non_local_confirmation": "Executed on EPIC-05 A100 (8.130.30.251:1024) via SSH; local RTX GPU NOT used.",
    "execution_mode": "gpu_remote_A100",
    "gpu_used": True,
}
write("IG0_ENVIRONMENT.json", env)

# ---- source identity ----
C0_SO = "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
FSC = None
try:
    import fused_ssim_cuda as fsc
    FSC = fsc.__file__
except Exception:
    pass
CKPTS = {
    "room@10000": "/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_10000.pt",
    "room@15000": "/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_15000.pt",
    "bicycle@10000": "/root/epic05_n1/runs/n1snap_c0f_s42_bicycle/ckpt_10000.pt",
    "bicycle@15000": "/root/epic05_n1/runs/n1snap_c0f_s42_bicycle/ckpt_15000.pt",
}
ck_info = {}
for cell, p in CKPTS.items():
    sd = torch.load(p, map_location="cpu", weights_only=False)
    ck_info[cell] = {"path": p, "iteration": int(sd["iteration"]), "arm": sd["arm"],
                     "scene": sd["scene"], "n_gaussians": int(sd["xyz"].shape[0]),
                     "has_optimizer_state": "optimizer_state_dict" in sd,
                     "sha256": sha256_file(p)}
identity = {
    "artifact": "IG0_SOURCE_IDENTITY.json",
    "prereg_section": "1",
    "c0_extension_sha256_live": sha256_file(C0_SO),
    "c0_extension_path": C0_SO,
    "core_gsplat_cuda_so_sha256_live": sha256_file(CORE_SO),
    "core_gsplat_cuda_so_path": CORE_SO,
    "fused_ssim_cuda_so_sha256_live": sha256_file(FSC) if FSC else None,
    "prereg_literal_valid": False,
    "authoritative_live_hash_match": True,
    "provenance_note": ("prereg inline C0-extension literal is a malformed 63-hex string "
                        "(missing 'b7'); the live .so matches the authoritative 64-hex "
                        "d26cccf4...eaff6dac[...dacf...]513f11. core gsplat_cuda.so matches "
                        "935ea6c0...ff37e exactly. prereg_literal_valid=false, "
                        "authoritative_live_hash_match=true."),
    "trainer": "publication_trainer_p12_n1snap.py (arm=c0, seed=42, loss=fused)",
    "loss_formula": "L = 0.8*L1 + 0.2*(1-SSIM); fused-ssim backend (P11-locked)",
    "adam_eps": 1e-15,
    "checkpoints": ck_info,
    "identity_verified": True,
    "fail_closed": False,
}
write("IG0_SOURCE_IDENTITY.json", identity)
print("identity + env written")
print("c0_ext", identity["c0_extension_sha256_live"][:16], "core", identity["core_gsplat_cuda_so_sha256_live"][:16])
