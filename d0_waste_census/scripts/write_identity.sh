#!/bin/bash
# D0 Stage 0a: frozen-identity audit -> /root/epic05_d0/artifacts/D0_SOURCE_IDENTITY.json
set -euo pipefail
ART=/root/epic05_d0/artifacts
SRC=/root/epic05_d0/scripts
WT=/root/epic05_d0/c0_worktree_d0
mkdir -p "$ART"

python3 - "$ART" "$SRC" "$WT" <<'PY'
import hashlib, json, os, sys

art, src, wt = sys.argv[1], sys.argv[2], sys.argv[3]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def rec(label, path, want=None):
    h = sha(path)
    return {"label": label, "path": path, "sha256": h, "expect": want,
            "match": (want is None or h == want)}

trainer = "/root/epic05_p12/publication_trainer_p12.py"
c0_ext = ("/root/epic05_build_cache/c0_ext/"
          "experimental_gaussian_render_inference_scene_cuda/"
          "experimental_gaussian_render_inference_scene_cuda.so")
c0_core = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
fused_candidates = []
import glob
for p in sorted(glob.glob("/root/miniforge3/envs/gsplat/lib/python*/site-packages/"
                          "fused_ssim_cuda*.so")):
    if os.path.isfile(p):
        fused_candidates.append(rec("fused_ssim_cuda_so", p,
                                    "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"))

items = [
    rec("frozen_trainer_p12", trainer,
        "675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5"),
    rec("c0_ext_so", c0_ext,
        "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11"),
    rec("c0_core_so", c0_core,
        "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"),
    rec("frozen_renderer_py",
        "/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/"
        "render/functional/gaussian_inference.py",
        "d599aae3c1864993934d65646bfcb33e07e6046e389f028230752200df0c3a8a"),
    rec("repo_gaussian_model_py",
        "/root/3dgs-renderer-benchmark/baseline/reference_v1/gaussian_model.py",
        "68731e375013a6f24aa4d81e5986402b1ac50f51cddf20688e358be705b1a60b"),
    rec("d0_worktree_renderer_py",
        os.path.join(wt, "gsplat/experimental/render/functional/"
                          "gaussian_inference.py")),
    rec("d0_trainer_py", os.path.join(src, "d0_trainer.py")),
    rec("d0_gaussian_model_py", os.path.join(src, "d0_gaussian_model.py")),
    rec("d0_trace_py", os.path.join(src, "d0_trace.py")),
    rec("patch_renderer_py", os.path.join(src, "patch_renderer.py")),
]

doc = {
    "stage": "0a",
    "date": "2026-09-28",
    "host": "EPIC-05",
    "prereg_pin_note": (
        "D0 follows the prereg pair (the binaries the P12A/P12B numbers were "
        "produced with). The certification lane's alternative mx-prebuilt pair "
        "9baf8655…/361b216b… is recorded, not re-chosen (see "
        "p12_support_audit/certification/P12_SOURCE_CERTIFICATION_REPORT.md)."
    ),
    "fused_ssim_so": fused_candidates,
    "artifacts": items,
    "all_match": all(i["match"] for i in items)
                 and bool(fused_candidates)
                 and all(f["match"] for f in fused_candidates),
}
assert doc["all_match"], "identity audit FAILED: " + json.dumps(
    [i for i in items if not i["match"]], indent=2)
out = os.path.join(art, "D0_SOURCE_IDENTITY.json")
with open(out, "w") as f:
    json.dump(doc, f, indent=2)
print("wrote", out, "all_match=True")
for i in items:
    print("  ", i["label"], i["sha256"][:16], "OK" if i["match"] else "MISMATCH")
PY
