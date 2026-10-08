#!/usr/bin/env python3
"""Regenerate points3D.ply from points3D.bin using the method repo's own loader code
path (scene.dataset_readers.read_points3D_binary + storePly), i.e. the exact bytes the
trainer would have written when converting bin -> ply itself."""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True, help="method source root, e.g. .../src/FastGS")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.repo_root)
    os.chdir(args.repo_root)
    from scene.dataset_readers import read_points3D_binary, storePly

    xyz, rgb, _ = read_points3D_binary(args.bin)
    storePly(args.out, xyz, rgb)
    print(f"WROTE {args.out} n_points={xyz.shape[0]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
