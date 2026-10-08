#!/usr/bin/env python3
"""Generate or verify the frozen P8 C0 renderer-source manifest.

Mode generate (run inside the frozen publication environment, outside the
renderer checkout): hashes the normative renderer-relevant file set plus a
full non-transient tree index.

Mode verify (run on a copy of the same tree): re-hashes every manifest entry
and recomputes the tree index, then compares against the remote index.

Deterministic by construction: no timestamps, LF line endings, lexicographic
order by relative_path. The manifest file bytes are what the P12 run guard
pins as expected_renderer_source_sha256.
"""
import argparse
import hashlib
import json
import os
import sys

CPP_SUFFIXES = {".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp"}
BUILD_CONFIG = ("pyproject.toml", "setup.py", "MANIFEST.in", ".gitmodules", "bootstrap.sh", "config.yaml")
EXCLUDE_PREFIXES = (
    "tests/", "docs/", "examples/", "profiling/", "lint/", ".github/", "assets/",
    "gsplat/cuda/csrc/third_party/glm/test/", "third_party/googletest/",
)
SKIP_DIRS = {".git", "__pycache__", "build"}
TRANSIENT_SUFFIXES = {".so", ".o", ".pyc", ".npy", ".pt", ".pth", ".log"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def walk_tree(root):
    """All non-transient regular files, relative posix paths, sorted."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name == ".git":
                continue
            if os.path.splitext(name)[1].lower() in TRANSIENT_SUFFIXES:
                continue
            full = os.path.join(dirpath, name)
            if not os.path.isfile(full):
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            found.append(rel)
    return sorted(found)


def in_manifest_set(rel):
    if any(rel.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return False
    if rel in BUILD_CONFIG:
        return True
    if rel.startswith("gsplat/") and rel.endswith(".py"):
        return True
    return os.path.splitext(rel)[1].lower() in CPP_SUFFIXES


def entries_for(root, relpaths):
    out = []
    for rel in relpaths:
        full = os.path.join(root, rel)
        out.append({"relative_path": rel, "size_bytes": os.path.getsize(full), "sha256": sha256_file(full)})
    return out


def tree_digest(files):
    lines = ["%s  %d  %s" % (f["sha256"], f["size_bytes"], f["relative_path"]) for f in files]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=1, ensure_ascii=True)
        stream.write("\n")


def generate(args):
    root = os.path.abspath(args.root)
    all_files = walk_tree(root)
    manifest_files = entries_for(root, [rel for rel in all_files if in_manifest_set(rel)])
    index_files = entries_for(root, all_files)
    manifest = {
        "schema": "p8_c0_frozen_source_manifest/v1",
        "root": root,
        "selection": {
            "included": [
                "CUDA/C++ sources and headers: *.cu *.cuh *.c *.cc *.cpp *.h *.hpp",
                "Python renderer package: gsplat/**/*.py",
                "build/config: " + ", ".join(BUILD_CONFIG),
            ],
            "excluded_prefixes": list(EXCLUDE_PREFIXES),
            "excluded_transient": "build outputs, caches, logs, checkpoints: dirs .git/__pycache__/build and suffixes " + " ".join(sorted(TRANSIENT_SUFFIXES)),
        },
        "n_files": len(manifest_files),
        "tree_n_files": len(index_files),
        "tree_digest": tree_digest(index_files),
        "tree_digest_rule": "sha256 over sorted lines '<sha256>  <size_bytes>  <relative_path>\\n'",
        "files": manifest_files,
    }
    write_json(args.out, manifest)
    write_json(args.index_out, {
        "schema": "p8_c0_frozen_source_tree_index/v1",
        "root": root,
        "tree_digest": manifest["tree_digest"],
        "n_files": len(index_files),
        "files": index_files,
    })
    print("manifest: %s (%d files)" % (args.out, len(manifest_files)))
    print("index:    %s (%d files)" % (args.index_out, len(index_files)))
    print("manifest_sha256:", sha256_file(args.out))
    print("tree_digest:", manifest["tree_digest"])
    return 0


def verify(args):
    root = os.path.abspath(args.tree)
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    remote_index = json.load(open(args.index, encoding="utf-8"))
    missing, mismatch, ok = [], [], 0
    for entry in manifest["files"]:
        full = os.path.join(root, entry["relative_path"])
        if not os.path.isfile(full):
            missing.append(entry["relative_path"])
            continue
        actual = sha256_file(full)
        if actual != entry["sha256"]:
            mismatch.append({"relative_path": entry["relative_path"], "expected": entry["sha256"], "actual": actual})
        else:
            ok += 1
    local_files = entries_for(root, walk_tree(root))
    remote_map = {f["relative_path"]: f for f in remote_index["files"]}
    local_map = {f["relative_path"]: f for f in local_files}
    only_remote = sorted(set(remote_map) - set(local_map))
    only_local = sorted(set(local_map) - set(remote_map))
    hash_mismatch = [
        {"relative_path": p, "expected": remote_map[p]["sha256"], "actual": local_map[p]["sha256"]}
        for p in sorted(set(remote_map) & set(local_map)) if remote_map[p]["sha256"] != local_map[p]["sha256"]
    ]
    local_digest = tree_digest(local_files)
    result = {
        "schema": "p12_c0_vs_p8_c0_renderer_diff/v1",
        "historical_tree": remote_index["root"],
        "p12_tree": root,
        "manifest_file": os.path.abspath(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "historical_tree_digest": remote_index["tree_digest"],
        "p12_tree_digest": local_digest,
        "normative_file_set": {
            "n": len(manifest["files"]),
            "identical": ok,
            "missing_in_p12": missing,
            "hash_mismatch": mismatch,
        },
        "full_tree_index": {
            "n_historical": len(remote_map),
            "n_p12": len(local_map),
            "only_in_historical": only_remote,
            "only_in_p12": only_local,
            "hash_mismatch": hash_mismatch,
        },
        "classification": {
            "identical_renderer_source": not (missing or mismatch or only_remote or only_local or hash_mismatch),
            "unexpected_renderer_differences": (missing + [m["relative_path"] for m in mismatch] + only_remote + only_local + [m["relative_path"] for m in hash_mismatch]),
        },
    }
    result["status"] = "IDENTICAL_RENDERER_SOURCE" if result["classification"]["identical_renderer_source"] else "RENDERER_SOURCE_DRIFT"
    write_json(args.out, result)
    print("status:", result["status"])
    print("normative: %d/%d identical, missing=%d mismatch=%d" % (ok, len(manifest["files"]), len(missing), len(mismatch)))
    print("full index: historical=%d p12=%d only_hist=%d only_p12=%d hash_mismatch=%d" % (
        len(remote_map), len(local_map), len(only_remote), len(only_local), len(hash_mismatch)))
    print("historical_tree_digest:", remote_index["tree_digest"])
    print("p12_tree_digest:       ", local_digest)
    return 0 if result["classification"]["identical_renderer_source"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--root", required=True)
    gen.add_argument("--out", required=True)
    gen.add_argument("--index-out", required=True)
    ver = sub.add_parser("verify")
    ver.add_argument("--tree", required=True)
    ver.add_argument("--manifest", required=True)
    ver.add_argument("--index", required=True)
    ver.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.mode == "generate":
        return generate(args)
    return verify(args)


if __name__ == "__main__":
    sys.exit(main())
