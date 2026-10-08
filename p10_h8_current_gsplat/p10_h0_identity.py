#!/usr/bin/env python3
"""P10 Gate H0 - source identity freeze.

Assembles the frozen-source record for the P10 H8-MR experiment:
upstream SHA, patch diff + SHA256, per-file content hashes for both arms,
produced .so SHA256s, compiler flags, toolchain/env identity, GPU identity.

Runs on the build host (reads /root/epic05_p10).
"""
import argparse
import hashlib
import json
import os
import subprocess

ROOT = "/root/epic05_p10"
PATCH = f"{ROOT}/P10_H8_CURRENT_GSPLAT_PATCH.diff"
UPSTREAM_SHA = "512d366b67073d77ca099ede742683c165dfc23b"
MODIFIED = [
    "gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu",
    "gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu",
    "gsplat/cuda/include/Common.h",
]
# Expected upstream blob hashes at the frozen commit, computed locally from the
# clone at 512d366b (git -c core.autocrlf=false show <rev>:<path>). The pristine
# tree on the build host must reproduce these exactly.
UPSTREAM_BLOB_SHA256 = {
    "gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu":
        "ff00a36c40813fbf7e2aa39c1a8c84378334b2e7282a09a4e3114aeae3e04ce0",
    "gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu":
        "00c7ef050aed6f79b6c6938b9a537da2ba23a74893dab6e95eee12c0ab2a120e",
    "gsplat/cuda/include/Common.h":
        "31cefccf8159caa862139f934926b1c3bec6699a392a4108babea63f44294d08",
}
# sha256 over sorted "relpath\0blob_sha256\n" for every upstream gsplat source
# file (282 files; third_party/glm excluded, pinned separately) -- a single value
# that pins the whole source tree the two arms were built from
UPSTREAM_SOURCE_MANIFEST_SHA256 = "d512240db45b2ebbb944ae6394ae4412e7ec7bae62514e5d37b07de0a1ab3a8f"
GLM_COMMIT = "2d4c4b4dd31fde06cfffad7915c2b3006402322f"
MANIFEST_EXTS = (".cu", ".cuh", ".h", ".hpp", ".cpp", ".py", ".pyi", ".txt", ".md")


def source_manifest(tree, prefix="gsplat"):
    """Manifest hash of a built tree's source files, matching the upstream one."""
    base = os.path.join(tree, prefix)
    entries = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "third_party", ".git")]
        for fn in filenames:
            if not fn.endswith(MANIFEST_EXTS):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, tree).replace(os.sep, "/")
            entries.append(f"{rel}\0{sha256_file(p)}\n")
    entries.sort()
    return hashlib.sha256("".join(entries).encode()).hexdigest(), len(entries)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = {}
    for cmd, key in [("hostname", "hostname"), ("uname -sr", "uname"),
                     ("nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader", "gpus"),
                     ("nvidia-smi --query-gpu=compute_cap --format=csv,noheader", "compute_caps"),
                     ("cat /proc/cpuinfo | grep -m1 'model name'", "cpu")]:
        env[key] = sh(cmd)
    python_env = json.loads(sh(
        "/root/miniforge3/envs/gsplat/bin/python -c \""
        "import json,torch,sys;print(json.dumps({'python':sys.version.split()[0],"
        "'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'cudnn':torch.backends.cudnn.version()}))\"" ) or "{}")
    nvcc = sh("/root/miniforge3/envs/gsplat/bin/nvcc --version | tail -1")

    patch = {"path": PATCH, "sha256": sha256_file(PATCH),
             "bytes": os.path.getsize(PATCH),
             "lines": sum(1 for _ in open(PATCH)),
             "files_touched_declared": MODIFIED,
             "files_touched_in_diff": sorted({
                 l.split(" b/", 1)[1].strip() for l in open(PATCH)
                 if l.startswith("+++ b/")}),
             "patch_is_isolated": None}

    arms = {}
    for arm in ("pristine", "h8"):
        tree = f"{ROOT}/tree_{arm}"
        man, n = source_manifest(tree)
        arms[arm] = {
            "tree": tree,
            "modified_file_sha256": {f: sha256_file(os.path.join(tree, f)) for f in MODIFIED},
            "source_manifest_sha256": man,
            "source_manifest_n_files": n,
            "matches_upstream_manifest": man == UPSTREAM_SOURCE_MANIFEST_SHA256,
        }
        build_json = f"{ROOT}/build_identity_{arm}.json"
        if os.path.exists(build_json):
            with open(build_json) as f:
                b = json.load(f)
            arms[arm]["extension"] = {
                "name": b["extension_name"], "so_path": b["so_path"],
                "so_sha256": b["so_sha256"], "so_size": b["so_size"],
                "n_sources": b["n_sources"],
                "source_manifest_sha256": {k: v for k, v in sorted(b["source_manifest"].items())},
                "build_flags": b["build_flags"],
                "build_ninja_sha256": b["build_ninja_sha256"],
                "nvcc_command_example": b["nvcc_command_example"],
                "env": b["env"], "versions": b["versions"],
            }

    # per-file upstream provenance of the pristine tree: the three files the
    # patch touches must be byte-identical to the upstream blobs at 512d366b
    pristine_provenance = {}
    for f in MODIFIED:
        got = sha256_file(os.path.join(f"{ROOT}/tree_pristine", f))
        pristine_provenance[f] = {
            "sha256": got,
            "expected_upstream_sha256": UPSTREAM_BLOB_SHA256[f],
            "matches_upstream_blob": got == UPSTREAM_BLOB_SHA256[f],
        }

    # patch isolation: exactly the declared files, and nothing else in the tree
    # changed (manifest of h8 vs pristine differs only in those 3 files)
    diff_files = sorted({
        l.split(" b/", 1)[1].strip() for l in open(PATCH) if l.startswith("+++ b/")})
    patch["patch_is_isolated"] = diff_files == sorted(MODIFIED)
    shared = set(arms["pristine"]["modified_file_sha256"]) & set(arms["h8"]["modified_file_sha256"])
    changed = sorted(f for f in shared
                     if arms["pristine"]["modified_file_sha256"][f]
                     != arms["h8"]["modified_file_sha256"][f])

    # patch applicability + isolation evidence
    apply_check = sh(f"cd {ROOT}/tree_pristine && patch -p1 --dry-run < {PATCH}")
    apply_rc = subprocess.run(f"cd {ROOT}/tree_pristine && patch -p1 --dry-run --silent < {PATCH}",
                              shell=True).returncode
    tree_diff = [l for l in sh(f"diff -r -q {ROOT}/tree_pristine/gsplat {ROOT}/tree_h8/gsplat").splitlines()
                 if l.strip() and "__pycache__" not in l]

    record = {
        "phase": "P10 H8-MR on current gsplat main",
        "upstream": {
            "repository": "nerfstudio-project/gsplat",
            "commit": UPSTREAM_SHA,
            "glm_submodule_commit": "2d4c4b4dd31fde06cfffad7915c2b3006402322f",
        },
        "patch": patch,
        "pristine_tree_upstream_provenance": pristine_provenance,
        "pristine_tree_matches_upstream_manifest": arms["pristine"]["matches_upstream_manifest"],
        "h8_tree_matches_upstream_manifest": arms["h8"]["matches_upstream_manifest"],
        "source_manifest_sha256_upstream_expected": UPSTREAM_SOURCE_MANIFEST_SHA256,
        "files_changed_between_arms": changed,
        "patch_applies_cleanly_to_pristine_tree": {
            "dry_run_output": apply_check.splitlines(), "returncode": apply_rc,
            "applies_cleanly": apply_rc == 0,
        },
        "tree_diff_pristine_vs_h8_excluding_pycache": tree_diff,
        "arms": arms,
        "environment": {**env, "python_env": python_env, "nvcc": nvcc,
                        "conda_env": "/root/miniforge3/envs/gsplat",
                        "build_env_flags": {"TORCH_CUDA_ARCH_LIST": "8.0", "MAX_JOBS": "32"}},
        "gpu_used_for_measurements": {
            "cuda_visible_devices": "0",
            "note": "EPIC-05 GPUs 0-3 idle and used; GPUs 4-7 belong to an unrelated job",
        },
    }
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"H0 identity -> {args.out}")
    print(f"  patch sha256={patch['sha256'][:16]}... isolated={patch['patch_is_isolated']} "
          f"applies_cleanly={record['patch_applies_cleanly_to_pristine_tree']['applies_cleanly']}")
    print(f"  pristine matches upstream: files="
          f"{all(v['matches_upstream_blob'] for v in pristine_provenance.values())} "
          f"manifest={arms['pristine']['matches_upstream_manifest']}")
    print(f"  files changed between arms: {changed}")
    for arm in ("pristine", "h8"):
        so = arms[arm].get("extension", {}).get("so_sha256", "MISSING")
        print(f"  {arm}: so_sha256={so if so == 'MISSING' else so[:16] + '...'}")


if __name__ == "__main__":
    main()
