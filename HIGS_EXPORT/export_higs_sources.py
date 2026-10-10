#!/usr/bin/env python3
"""Read-only source handoff exporter for MX HiGS CUDA forensic work.

No imports from gsplat, no code execution, no source modifications.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

EXTS = {'.cu', '.cuh', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.py'}
SCAN_LIMIT_BYTES = 4 * 1024 * 1024
EXPORT_LIMIT_BYTES = 48 * 1024 * 1024
CRITICAL_BASENAMES = {'IntersectTile.cu', 'ext.cpp', 'setup.py'}
TOKENS = (b'rasterize_gaussian_higs_frozen', b'gsplat::intersect_tile', b'intersect_tile(', b'device_totals', b'n_isects')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_candidates(root):
    # Do not follow symlinks; exclude virtual envs and version-control directories.
    excluded = {'.git', 'build', 'dist', '__pycache__', '.venv', 'venv', 'node_modules'}
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in excluded and not Path(base, d).is_symlink()]
        for name in files:
            path = Path(base) / name
            if path.suffix.lower() not in EXTS or path.is_symlink():
                continue
            yield path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')
    ap.add_argument('--previous-handoff', default='/mnt/storage_pool/liaoyuanjun/HIGS_SOURCE_HANDOFF')
    ap.add_argument('--r8', default='/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_RUNTIME_RECOVERY_R8_MX')
    ap.add_argument('--out', default='/mnt/storage_pool/liaoyuanjun/HIGS_SOURCE_HANDOFF_ACTUAL')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    if not root.is_dir():
        raise SystemExit(f'MISSING_ROOT: {root}')
    if out == root or root in out.parents:
        raise SystemExit('Output must NOT be within the source root')
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'Output folder must be new/empty; refusing overwrite: {out}')
    out.mkdir(parents=True, exist_ok=True)

    selected = []
    for path in safe_candidates(root):
        try:
            size = path.stat().st_size
            if size > SCAN_LIMIT_BYTES:
                continue
            relative = path.relative_to(root)
            chosen = path.name in CRITICAL_BASENAMES
            chosen |= 'experimental/render/' in str(relative).replace('\\', '/')
            if not chosen:
                content = path.read_bytes()
                chosen = any(token in content for token in TOKENS)
            if chosen:
                selected.append((path, 'source', str(relative)))
        except (OSError, UnicodeError) as exc:
            print(f'SKIPPED {path}: {exc}', file=sys.stderr)

    external = [
        (Path(args.previous_handoff), 'previous_handoff', ['IntersectTile.cu', 'HIGS_PYTHON_WRAPPER.py', 'HIGS_BINDINGS.cpp', 'R8_WORKING_ENTRYPOINT.py', 'COUNTING_CALL_GRAPH.md', 'BUILD_CONFIGURATION.txt', 'RUNTIME_BINARY_IDENTITY.json']),
        (Path(args.r8), 'r8', ['R8_WORKING_ENTRYPOINT.py', 'R8_RUNTIME_ENTRYPOINT.md', 'R8_R3_ORIGINAL_COMMAND.txt']),
    ]
    for directory, category, names in external:
        if directory.is_dir():
            for name in names:
                hits = list(directory.rglob(name))
                for hit in hits[:4]:
                    if hit.is_file() and not hit.is_symlink() and hit.stat().st_size < SCAN_LIMIT_BYTES:
                        selected.append((hit, category, str(hit.relative_to(directory))))

    # Deduplicate by output path and avoid copying more than reasonable source size.
    manifest = []
    used = set()
    budget = EXPORT_LIMIT_BYTES
    for path, category, relative in sorted(selected, key=lambda x: (x[1], x[2])):
        dest_rel = Path(category) / relative
        if str(dest_rel) in used:
            continue
        used.add(str(dest_rel))
        size = path.stat().st_size
        if size > budget:
            print(f'SKIPPED_BUDGET {path}', file=sys.stderr)
            continue
        dest = out / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        budget -= size
        manifest.append({'file': str(dest_rel), 'source_path': str(path), 'bytes': size, 'sha256': sha256(dest)})

    binaries = []
    for name in ['gsplat/csrc.so', 'gsplat/experimental/render/kernels/csrc.so']:
        path = root / name
        if path.is_file():
            binaries.append({'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size})
    missing = []
    have = {Path(x['file']).name for x in manifest}
    for name in ['IntersectTile.cu','setup.py']:
        if name not in have: missing.append(name)
    has_entry = any('rasterize_gaussian_higs_frozen' in (out / x['file']).read_text(errors='replace') for x in manifest if Path(x['file']).suffix in EXTS)
    if not has_entry: missing.append('rasterize_gaussian_higs_frozen implementation')

    summary = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_root': str(root),
        'source_count': len(manifest),
        'binary_hashes': binaries,
        'missing_required': missing,
        'files': manifest,
        'note': 'Export is read-only. Source provenance is not established by this archive alone; compare build records and loaded .so separately.',
    }
    (out/'SOURCE_MANIFEST.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    (out/'SHA256SUMS.txt').write_text(''.join(f'{x["sha256"]}  {x["file"]}\n' for x in manifest))
    archive = out.parent / (out.name + '.tar.gz')
    if archive.exists():
        raise SystemExit(f'Refusing overwrite: {archive}')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(out, arcname=out.name, recursive=True)
    print(json.dumps({'archive': str(archive), 'archive_sha256': sha256(archive), 'files': len(manifest), 'missing': missing, 'binaries': binaries}, indent=2))
    if missing:
        print('WARNING: Required files missing, source handoff is incomplete.', file=sys.stderr)
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())
