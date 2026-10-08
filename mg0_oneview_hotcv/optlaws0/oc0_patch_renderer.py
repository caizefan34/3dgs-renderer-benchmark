#!/usr/bin/env python3
"""OPTLAWS0 / OC0 step 3 — build the additive diagnostic renderer copy.

Usage: oc0_patch_renderer.py [source_worktree] [dest_worktree]

Copies the frozen C0 worktree to a NEW directory and inserts ONE additive,
metadata-only hook into

    gsplat/experimental/render/functional/gaussian_inference.py

The hook publishes the tensors that `_HigsAutogradFunction._native_forward_capture`
has ALREADY computed (the `captured` tuple + `visible_ids` + the autograd ctx) to
the importing module.  It

  * adds no tensor, removes none, reorders no op, consumes no RNG;
  * runs entirely inside the existing `with torch.no_grad():` block;
  * does not touch the production worktree or any compiled .so.

The frozen worktree is never modified.  The generated diff is written next to
this script as OC0_RENDERER_DIFF.diff and the identity of both trees is recorded
in OC0_PATCH.json.
"""
import ast
import difflib
import hashlib
import json
import os
import shutil
import sys
import time

import oc0_common as K

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "gsplat/experimental/render/functional/gaussian_inference.py"

DEFAULT_SRC = "/root/epic05_frozen/higs_c0_final30k_worktree"
DEFAULT_DST = "/root/epic05_mg0/c0_worktree_oc0"

ANCHOR = "        return render_colors, render_alphas, captured\n"
INSERT = (
    "        # OC0 instrumentation (additive exposure only): publish the tensors\n"
    "        # this native forward has already computed, for the diagnostic\n"
    "        # replay of OC0_PREREG.  No value, op order or RNG stream changes.\n"
    "        import sys as _oc0_sys\n"
    "        _oc0_sys.modules[__name__]._OC0_LAST = {\n"
    "            \"captured\": captured,\n"
    "            \"visible_ids\": visible_ids,\n"
    "            \"ctx\": ctx,\n"
    "        }\n"
    "        return render_colors, render_alphas, captured\n"
)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tree_digest(root):
    """Deterministic digest of the *.py files under root (sorted, relative)."""
    h = hashlib.sha256()
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            h.update(os.path.relpath(p, root).replace(os.sep, "/").encode())
            h.update(open(p, "rb").read())
            n += 1
    return h.hexdigest(), n


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    dst = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DST
    t0 = time.time()
    if not os.path.isdir(os.path.join(src, os.path.dirname(REL))):
        raise SystemExit(f"source worktree not found: {src}")
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("source and destination must differ")

    if os.path.isdir(dst):
        print(f"[oc0_patch] removing stale {dst}", flush=True)
        shutil.rmtree(dst)
    shutil.copytree(src, dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))

    path = os.path.join(dst, REL)
    orig = open(path, encoding="utf-8").read()
    n = orig.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"anchor occurs {n} times (must be exactly 1)")
    if "_OC0_LAST" in orig:
        raise SystemExit("destination already instrumented")
    text = orig.replace(ANCHOR, INSERT, 1)
    ast.parse(text)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    diff = "".join(difflib.unified_diff(
        orig.splitlines(keepends=True), text.splitlines(keepends=True),
        fromfile="frozen/" + REL, tofile="oc0/" + REL, n=3))
    os.makedirs(K.ARTIFACTS, exist_ok=True)
    with open(K.artifact("OC0_RENDERER_DIFF.diff"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(diff)

    src_d, src_n = tree_digest(src)
    dst_d, dst_n = tree_digest(dst)
    out = {
        "campaign": "OPTLAWS0", "stage": "OC0_patch_renderer",
        "creation_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_worktree": src, "dest_worktree": dst,
        "patched_file": REL,
        "insertions": 1, "deletions": 0,
        "diff_lines": len(diff.splitlines()),
        "frozen_file_sha256": hashlib.sha256(orig.encode()).hexdigest(),
        "patched_file_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "frozen_tree_py_digest": src_d, "frozen_tree_py_files": src_n,
        "patched_tree_py_digest": dst_d, "patched_tree_py_files": dst_n,
        "additive_only": True,
        "note": "One insertion, no deletion; the inserted block only publishes "
                "already-computed tensors through sys.modules[__name__] and runs "
                "inside the existing torch.no_grad() region.",
        "script_sha256": {os.path.basename(__file__): sha256_file(os.path.abspath(__file__))},
    }
    with open(K.artifact("OC0_PATCH.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[oc0_patch] built {dst} (+{len(diff.splitlines())} diff lines) "
          f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()