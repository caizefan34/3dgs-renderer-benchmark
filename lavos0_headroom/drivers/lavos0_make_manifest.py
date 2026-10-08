#!/usr/bin/env python
"""Build MANIFEST.sha256 for a LAVOS0 output directory (DSH handoff step 10).

Format matches the existing repo convention (reprotopo0_final/MANIFEST.sha256):
    <sha256>  <filename>
one line per artifact, sorted by filename. Verifiable with:
    sha256sum -c MANIFEST.sha256
"""

import argparse
import hashlib
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(outdir, files=None, manifest_name="MANIFEST.sha256"):
    """Write MANIFEST.sha256 over `files` (default: all files except the
    manifest itself). Returns the list of (sha256, name) lines written."""
    if files is None:
        files = sorted(
            f for f in os.listdir(outdir)
            if os.path.isfile(os.path.join(outdir, f)) and f != manifest_name
        )
    lines = []
    for name in sorted(files):
        path = os.path.join(outdir, name)
        if not os.path.isfile(path):
            continue
        lines.append((sha256_file(path), name))
    manifest_path = os.path.join(outdir, manifest_name)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        for digest, name in lines:
            f.write(f"{digest}  {name}\n")
    return lines


def build_manifest_recursive(outdir, manifest_name="MANIFEST.sha256",
                           skip_dirs=("__pycache__", "a100_out"),
                           skip_files=("MANIFEST.sha256",)):
    lines = []
    for dirpath, dirnames, filenames in os.walk(outdir):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            if fn in skip_files:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, outdir).replace(os.sep, "/")
            lines.append((sha256_file(full), rel))
    manifest_path = os.path.join(outdir, manifest_name)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        for digest, rel in sorted(lines, key=lambda x: x[1]):
            f.write(f"{digest}  {rel}\n")
    return lines


def verify_manifest(outdir, manifest_name="MANIFEST.sha256"):
    """Verify MANIFEST.sha256. Returns (ok, problems)."""
    manifest_path = os.path.join(outdir, manifest_name)
    if not os.path.isfile(manifest_path):
        return False, ["manifest missing"]
    problems = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                problems.append(f"bad line: {line!r}")
                continue
            digest, name = parts
            path = os.path.join(outdir, name)
            if not os.path.isfile(path):
                problems.append(f"missing file: {name}")
                continue
            actual = sha256_file(path)
            if actual != digest:
                problems.append(f"hash mismatch: {name}")
    return (len(problems) == 0), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--recursive", action="store_true",
                    help="walk outdir recursively with relative paths")
    args = ap.parse_args()
    if args.verify:
        ok, problems = verify_manifest(args.outdir)
        print("MANIFEST OK" if ok else f"MANIFEST FAIL: {problems}")
        sys.exit(0 if ok else 1)
    if args.recursive:
        lines = build_manifest_recursive(args.outdir)
    else:
        lines = build_manifest(args.outdir)
    print(f"wrote MANIFEST.sha256 with {len(lines)} entries")


if __name__ == "__main__":
    main()
