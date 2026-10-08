#!/usr/bin/env python3
"""P12 support: verify frozen-renderer source manifests on disk.

Checks, for each build-identity record, that every source file listed in the
identity's source_manifest still hashes to the recorded sha256, and computes
aggregate source manifests for the C0 worktree.

Read-only. Emits JSON on stdout.
"""
import hashlib
import json
import os
import subprocess
import sys


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_identity(ident_path):
    ident = json.load(open(ident_path))
    tree = ident.get("source_tree")
    manifest = ident.get("source_manifest") or {}
    missing, mismatch, ok = [], [], 0
    for rel, want in sorted(manifest.items()):
        p = os.path.join(tree, rel)
        if not os.path.exists(p):
            missing.append(rel)
            continue
        got = sha256_file(p)
        if got != want:
            mismatch.append({"file": rel, "expected": want, "actual": got})
        else:
            ok += 1
    return {
        "identity_file": ident_path,
        "artifact": ident.get("artifact"),
        "so_sha256_epic05": ident.get("so_sha256_epic05"),
        "source_tree": tree,
        "n_sources": len(manifest),
        "verified_ok": ok,
        "missing": missing,
        "mismatches": mismatch,
        "all_match": (len(missing) == 0 and len(mismatch) == 0),
    }


def aggregate_manifest(tree, roots, patterns, exclude=("third_party", "__pycache__")):
    cmd = (f"cd {tree} && find {' '.join(roots)} -type f "
           + r"\( " + " -o ".join(f"-name '{p}'" for p in patterns) + r" \) "
           + "| grep -v " + " | grep -v ".join(exclude)
           + " | sort | xargs sha256sum | sha256sum")
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.strip()
    n = subprocess.run(["bash", "-c", cmd.rsplit("|", 2)[0] + "| wc -l"],
                       capture_output=True, text=True).stdout.strip()
    return {"tree": tree, "aggregate_sha256": out.split()[0] if out else None,
            "n_files": int(n) if n else None, "command": cmd}


def newest_mtime(tree):
    out = subprocess.run(
        ["bash", "-c", f"find {tree} -type f -name '*.py' -printf '%T@ %p\\n' | sort -rn | head -3"],
        capture_output=True, text=True).stdout.strip()
    return out.splitlines()


def main():
    rec = {"host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()}
    rec["c0_ext_identity"] = verify_identity(
        "/root/3dgs-publication-epic05/provenance/"
        "epic05_build_identity_experimental_gaussian_render_inference_scene_cuda.json")
    rec["core_identity"] = verify_identity(
        "/root/3dgs-publication-epic05/provenance/epic05_build_identity_gsplat_cuda.json")
    c0wt = "/root/epic05_frozen/higs_c0_final30k_worktree"
    rec["c0_worktree_python_manifest"] = aggregate_manifest(c0wt, ["gsplat"], ["*.py"])
    rec["c0_worktree_full_source_manifest"] = aggregate_manifest(
        c0wt, ["."], ["*.py", "*.cu", "*.cuh", "*.cpp", "*.h", "*.hpp"])
    rec["c0_worktree_newest_python_files"] = newest_mtime(c0wt)
    gmain = "/root/epic05_p8/gsplat-main-512d366b"
    rec["gmain_tree_manifest"] = aggregate_manifest(
        gmain, ["gsplat"], ["*.py", "*.cu", "*.cuh", "*.cpp", "*.h", "*.hpp"])
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
