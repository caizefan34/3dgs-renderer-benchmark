#!/bin/bash
# P13-X setup: create /root/epic05_p13x/, copy P12 trainer UNTOUCHED,
# generate P13X_SOURCE_IDENTITY.json BEFORE any behavior modification.
set -eu
CAMP=/root/epic05_p13x
mkdir -p "$CAMP/artifacts" "$CAMP/logs" "$CAMP/runs"
cp -p /root/epic05_p12/publication_trainer_p12.py "$CAMP/publication_trainer_p13x.py"
chmod +w "$CAMP/publication_trainer_p13x.py"

PY=/root/miniforge3/envs/gsplat/bin/python
"$PY" - <<'EOF'
import hashlib, json, os, subprocess, datetime

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

CAMP = "/root/epic05_p13x"
ident = {
    "phase": "P13-X",
    "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "purpose": "Source identity snapshot taken BEFORE any behavioral modification of the copied trainer (P13X_DSH_TASK.md first action).",
    "baseline_trainer": {
        "path": "/root/epic05_p12/publication_trainer_p12.py",
        "sha256": sha256_file("/root/epic05_p12/publication_trainer_p12.py"),
        "modified_during_setup": False,
    },
    "p13x_trainer_copy": {
        "path": os.path.join(CAMP, "publication_trainer_p13x.py"),
        "sha256": sha256_file(os.path.join(CAMP, "publication_trainer_p13x.py")),
        "byte_identical_to_p12": True,
        "state": "pristine copy; behavioral edits come AFTER this file was recorded",
    },
    "frozen_binaries": {
        "c0_ext_sha256": sha256_file("/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"),
        "c0_core_sha256": sha256_file("/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"),
        "fused_ssim_so_sha256": sha256_file("/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so"),
        "prereg_expected": {
            "c0_ext": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
            "c0_core": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
            "fused_ssim": "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb",
        },
    },
    "env": {
        "python": subprocess.run(["/root/miniforge3/envs/gsplat/bin/python", "-c",
            "import sys, torch; print(sys.version.split()[0]); print(torch.__version__); print(torch.version.cuda)"],
            capture_output=True, text=True).stdout.split(),
        "hostname": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip().splitlines()[0],
    },
    "inputs_to_reuse": {
        "p12_campaign": "/root/epic05_p12 (trainer, artifacts, runs, manifests, contamination policy)",
        "frozen_worktree": "/root/epic05_frozen/higs_c0_final30k_worktree",
    },
}
ok = (ident["frozen_binaries"]["c0_ext_sha256"] == ident["frozen_binaries"]["prereg_expected"]["c0_ext"]
      and ident["frozen_binaries"]["c0_core_sha256"] == ident["frozen_binaries"]["prereg_expected"]["c0_core"]
      and ident["frozen_binaries"]["fused_ssim_so_sha256"] == ident["frozen_binaries"]["prereg_expected"]["fused_ssim"])
ident["frozen_binaries"]["match_prereg"] = bool(ok)
assert ident["baseline_trainer"]["sha256"] == ident["p13x_trainer_copy"]["sha256"], "copy is not byte-identical"
assert ok, "FROZEN BINARY HASH MISMATCH vs P13X_PREREG"
with open(os.path.join(CAMP, "artifacts", "P13X_SOURCE_IDENTITY.json"), "w") as f:
    json.dump(ident, f, indent=2)
print(json.dumps({"ok": True, "p12_sha256": ident["baseline_trainer"]["sha256"],
                  "binaries_match_prereg": ok}, indent=2))
EOF
echo "SETUP_DONE"
