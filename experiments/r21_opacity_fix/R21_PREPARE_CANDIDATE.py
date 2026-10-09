#!/usr/bin/env python3
"""Status: IMPLEMENTED. Create a NEW R21 root from the validated R20 root.
No compilation: both binaries and all CUDA/C++ sources remain byte identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REL = 'gsplat/experimental/render/functional/gaussian_inference.py'
MAIN = 'gsplat/csrc.so'
EXT = 'gsplat/experimental/render/kernels/csrc.so'
STD_SHA = '40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50'
R20_WRAPPER_SHA = '06ec8ed45f2ee38f21bab5e30b66c6b3c7298c85d0254e96cc00ed508879c1ae'
R20_EXT_SHA = 'b3c3e2e037f5c7f8921bc8b1e62c6bf924de88e23380a4bd195343c1146142ac'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def verify_delivery():
    manifest = json.loads((HERE/'R21_CANDIDATE_MANIFEST.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['files'].items():
        if sha(HERE/rel) != expected: raise ValueError(f'Delivery hash mismatch: {rel}')
    for rel, expected in manifest['dependencies'].items():
        # Repository text dependencies are normalized for Windows/Linux checkout.
        actual = hashlib.sha256((REPO/rel).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected: raise ValueError(f'Dependency hash mismatch: {rel}')
    return manifest


def verify_candidate(root):
    root = Path(root).resolve()
    manifest = verify_delivery()
    identity = json.loads((root/'R21_BUILD_IDENTITY.json').read_text())
    if sha(root/MAIN) != STD_SHA: raise ValueError('Frozen main binary mismatch')
    if sha(root/REL) != manifest['files']['source/'+REL]: raise ValueError('R21 wrapper mismatch')
    if sha(root/EXT) != R20_EXT_SHA or identity['r20_extension_sha256'] != R20_EXT_SHA:
        raise ValueError('Validated MX R20 extension changed')
    r20 = json.loads((root/'R20_BUILD_IDENTITY.json').read_text())
    # Use the same key as the unchanged R20 trainer.
    if sha(root/EXT) != r20['binary_sha256']: raise ValueError('R20 build identity mismatch')
    for rel, expected in identity['unchanged_files'].items():
        if sha(root/rel) != expected: raise ValueError(f'Candidate dependency changed: {rel}')
    return identity


def prepare(source, target):
    manifest = verify_delivery()
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists() or source == target or source in target.parents or target in source.parents:
        raise ValueError('Use a NEW, separate candidate directory outside the R20 root')
    if sha(source/MAIN) != STD_SHA or sha(source/REL) != R20_WRAPPER_SHA:
        raise ValueError('Input is not the audited R20 candidate')
    r20 = json.loads((source/'R20_BUILD_IDENTITY.json').read_text())
    if sha(source/EXT) != R20_EXT_SHA or r20['binary_sha256'] != R20_EXT_SHA:
        raise ValueError('Validated MX R20 binary identity mismatch')
    expected = json.loads((REPO/'experiments/r20_true_absgrad/R20_SOURCE_MANIFEST.json').read_text())
    for rel, entry in expected['files'].items():
        if sha(source/rel) != entry['patched_sha256']: raise ValueError(f'R20 source mismatch: {rel}')
    # Verify exact frozen projection/rendering/AccuTile dependencies before copying.
    frozen = json.loads((HERE/'evidence/MX_SOURCE_IDENTITY.json').read_text())
    for rel, entry in frozen['files'].items():
        if rel == EXT: continue  # R20 intentionally replaced the original HiGS binary.
        if sha(source/rel) != entry['sha256']: raise ValueError(f'Frozen dependency mismatch: {rel}')
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'build', '*.egg-info')
    # Copy files, never hardlink; target must be absent. Do not touch source metadata.
    shutil.copytree(source, target, ignore=ignore)
    shutil.copyfile(HERE/'source'/REL, target/REL)
    unchanged = {}
    for p in target.rglob('*'):
        if p.is_file() and p.relative_to(target).as_posix() != REL:
            rel = p.relative_to(target).as_posix()
            original = sha(source/rel)
            if sha(p) != original: raise ValueError(f'Unexpected copied-file difference: {rel}')
            unchanged[rel] = original
    identity = dict(status='PREPARED_NOT_GPU_VALIDATED', source_root=str(source), candidate_root=str(target),
        implementation_commit=manifest['implementation_commit'],
        r20_extension_sha256=sha(source/EXT), frozen_main_sha256=STD_SHA,
        r20_wrapper_sha256=R20_WRAPPER_SHA, r21_wrapper_sha256=sha(target/REL),
        unchanged_files=unchanged, delivery_manifest_sha256=sha(HERE/'R21_CANDIDATE_MANIFEST.json'))
    (target/'R21_BUILD_IDENTITY.json').write_text(json.dumps(identity, indent=2)+'\n')
    verify_candidate(target)
    return identity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--r20-root', type=Path); ap.add_argument('--candidate-root', type=Path)
    ap.add_argument('--verify', action='store_true'); ap.add_argument('--verify-delivery', action='store_true')
    args = ap.parse_args()
    if args.verify_delivery: result = verify_delivery()
    elif args.verify:
        if not args.candidate_root: ap.error('--candidate-root is required')
        result = verify_candidate(args.candidate_root)
    else:
        if not args.r20_root or not args.candidate_root: ap.error('Both root arguments are required')
        result = prepare(args.r20_root, args.candidate_root)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files', 'unchanged_files', 'dependencies')}, indent=2))


if __name__ == '__main__': main()
