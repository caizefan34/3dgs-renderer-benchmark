#!/usr/bin/env python3
"""P18-P0 Phase 0: verify frozen C0 identity + freeze the 8-camera set (N/8).

Runs on the A100 box. Writes:
  /root/epic05_p18/P18_SOURCE_IDENTITY.json
  /root/epic05_p18/P18_CAMERA_SET.json
Camera set is frozen BEFORE any timing/rendering (protocol section 7).
"""
import hashlib, json, math, os, sys, time
from pathlib import Path
import torch  # MUST import before loading any .so so libc10.so resolves

OUT = Path("/root/epic05_p18")
OUT.mkdir(parents=True, exist_ok=True)

FROZEN = {
    "p12_trainer": ("675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5",
                    "/root/epic05_p12/publication_trainer_p12.py"),
    "c0_ext_so": ("d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
                  "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"),
    "core_so": ("935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
                "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"),
    "fused_ssim_so": ("dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb",
                      "/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so"),
}
CKPT = {
    "room": ("17c1f5904472f24dbb86bc8f19318dc31341deb2f0d891a014662a49d75edc6b",
             "/root/epic05_p13x/runs/room_diag/ckpt_16000.pt"),
    "bicycle": ("0c36f5c7f127a23ba471e917bcf3ecbf6f9843c295c62bed50e1090aeb6d8890",
                "/root/epic05_p13x/runs/bicycle_diag/ckpt_16000.pt"),
}
REPO = "/root/3dgs-renderer-benchmark"
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
PHASE7 = os.path.join(REPO, "scripts", "epic05", "phase7")
SRC = os.path.join(REPO, "src")
REF_V1 = os.path.join(REPO, "baseline", "reference_v1")
N_PROBE = 8


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def jwrite(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    print(f"[p18] wrote {path}", flush=True)


def log(m):
    print(f"[p18 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_ext_module(name, so_path):
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def verify_identity():
    checks = {}
    ok = True
    for name, (expected, path) in FROZEN.items():
        ex = os.path.exists(path)
        h = sha256_file(path) if ex else None
        match = (h == expected)
        ok = ok and match
        checks[name] = {"path": path, "exists": ex, "sha256": h, "expected": expected, "match": match}
    cks = {}
    for scene, (exp, path) in CKPT.items():
        ex = os.path.exists(path)
        rec = {"path": path, "exists": ex, "sha256": None, "size_bytes": None}
        if ex:
            rec["sha256"] = sha256_file(path)
            rec["size_bytes"] = os.path.getsize(path)
            rec["match"] = (rec["sha256"] == exp)
        cks[scene] = rec
    ident = {
        "phase": "P18-P0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "frozen C0 identity snapshot before any P18-P0 work; P18-P0 consumes the frozen "
                   "C0 binaries and 16K checkpoints without modifying them. QSCAN variants are "
                   "compiled in a separate worktree copy; the frozen .so hashes above must still "
                   "verify (they are the DIRECT/baseline binaries).",
        "frozen_binaries": checks,
        "checkpoints_16k": cks,
        "c0_worktree": C0_WT,
        "scene_cuda_dir": SCENE_CUDA_DIR,
        "prereg": {"seed": 42, "tile": 16, "eps2d": 0.3,
                   "renderer": "classic C0 (rasterize_gaussian_higs_dynamic)",
                   "resolution": "1080p", "background": "black",
                   "HIGS env": {"HIGS_PX_RUNTIME": "2", "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint",
                                "HIGS_BWD_H8_MR": "1", "HIGS_BWD_ABSGRAD": "1"}},
        "verdict_identity": ("IDENTITY_VERIFIED" if ok else "ABORT_IDENTITY_MISMATCH"),
    }
    return ident, ok


class SceneCtx:
    def __init__(self, scene):
        os.environ["HIGS_PX_RUNTIME"] = "2"
        os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
        os.environ["HIGS_BWD_H8_MR"] = "1"
        os.environ["HIGS_BWD_ABSGRAD"] = "1"
        os.environ.pop("HIGS_DISABLE_F9", None)
        sys.path.insert(0, C0_WT)
        sys.path.insert(0, SCENE_CUDA_DIR)
        core = load_ext_module("gsplat_cuda", FROZEN["core_so"][1])
        sys.modules["gsplat.csrc"] = core
        exp = load_ext_module("experimental_gaussian_render_inference_scene_cuda", FROZEN["c0_ext_so"][1])
        sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
        sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
        sys.path.insert(0, PHASE7)
        sys.path.insert(0, SRC)
        sys.path.insert(0, REF_V1)
        from dataset import GTDataset
        self.dataset = GTDataset(scene=scene, repo_root=REPO, resolution="1080p",
                                 device="cuda", background="black")
        self.n_cameras = len(self.dataset)
        self.scene = scene
        self._cache = {}

    def camera_record(self, ci):
        if ci in self._cache:
            return self._cache[ci]
        import torch
        cam, gt = self.dataset.get_item(ci)
        rec = {
            "index": ci,
            "image_name": getattr(cam, "image_name", None) or getattr(cam, "img_name", None),
            "viewmatrix": cam.viewmatrix.detach().cpu().numpy().tolist(),
            "K": cam.K.detach().cpu().numpy().tolist(),
            "width": int(cam.image_width),
            "height": int(cam.image_height),
            "gt_sha256": sha256_bytes(gt.contiguous().cpu().numpy().tobytes()),
        }
        self._cache[ci] = rec
        return rec


def select_ids(n_cam):
    return [int(math.floor((k + 0.5) * n_cam / N_PROBE)) for k in range(N_PROBE)]


def main():
    ident, ok = verify_identity()
    jwrite(str(OUT / "P18_SOURCE_IDENTITY.json"), ident)
    if not ok:
        log("IDENTITY MISMATCH - aborting (protocol: do not proceed)")
        sys.exit(2)
    log("identity verified for all 4 frozen binaries + 2 checkpoints")

    camset = {"scenes": {},
              "algorithm": ("P18 section 7: sorted camera-ID quantiles, "
                            "j_k = floor((k+0.5)*N/8), k=0..7. P17 used N/12 "
                            "(different count) so this is a NEW P18 set; P17 "
                            "provenance recorded. Frozen BEFORE any timing."),
              "n_probe": N_PROBE,
              "frozen_utc": None}
    for scene in ("room", "bicycle"):
        ctx = SceneCtx(scene)
        ids = select_ids(ctx.n_cameras)
        records = {str(ci): ctx.camera_record(ci) for ci in sorted(set(ids))}
        camset["scenes"][scene] = {
            "n_cameras": ctx.n_cameras, "probe_ids": ids, "camera_records": records}
        log(f"{scene}: N={ctx.n_cameras} probes={ids}")
        del ctx
        torch.cuda.empty_cache()
    camset["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jwrite(str(OUT / "P18_CAMERA_SET.json"), camset)
    log("camera set frozen")


if __name__ == "__main__":
    main()
