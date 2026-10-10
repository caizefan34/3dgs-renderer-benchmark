#!/usr/bin/env python3
"""Status: IMPLEMENTED; create a NEW isolated full source tree from frozen MX.

Verify the exact four source preimages and frozen binaries BEFORE copying.
Copy the frozen std_ll binary; exclude only the old experimental binary.
Never install globally and never modify the frozen input directory.
"""
import argparse
import json
from pathlib import Path
import shutil
from R20_GRADIENT_PARITY_TESTS import STD_SHA, HIGS_SHA, sha, write_json

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-root', required=True); p.add_argument('--candidate-root', required=True)
    a = p.parse_args()
    frozen, candidate = Path(a.frozen_root).resolve(), Path(a.candidate_root).resolve()
    if candidate == frozen or frozen in candidate.parents or candidate in frozen.parents or candidate.exists():
        p.error('candidate must be a NEW separate directory outside frozen root')
    binaries = {'gsplat/csrc.so': STD_SHA, 'gsplat/experimental/render/kernels/csrc.so': HIGS_SHA}
    for rel, expected in binaries.items():
        if sha(frozen/rel) != expected:
            raise RuntimeError('frozen binary identity mismatch: '+rel)
    manifest = json.loads((HERE/'R20_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for rel, record in manifest['files'].items():
        if sha(frozen/rel) != record['original_sha256']:
            raise RuntimeError('source preimage mismatch; re-audit required: '+rel)
        if sha(HERE/'source'/rel) != record['patched_sha256']:
            raise RuntimeError('corrupt R20 overlay: '+rel)
    required = ['gsplat/cuda/include/Common.h', 'gsplat/cuda/include/Utils.cuh',
                'gsplat/cuda/csrc/Config.h', 'gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh',
                'gsplat/cuda/csrc/third_party/glm/glm/gtc/type_ptr.hpp', 'gsplat/_lazy_backend.py']
    for rel in required:
        if not (frozen/rel).is_file():
            raise RuntimeError('full MX source dependency missing: '+rel)
    def ignore(path, names):
        old_binary_dir = frozen/'gsplat/experimental/render/kernels'
        return ['csrc.so'] if Path(path) == old_binary_dir and 'csrc.so' in names else []
    shutil.copytree(frozen, candidate, ignore=ignore)
    for rel in manifest['files']:
        shutil.copyfile(HERE/'source'/rel, candidate/rel)
    # Verify the reference was unchanged by preparation.
    for rel, expected in binaries.items():
        assert sha(frozen/rel) == expected
    write_json(candidate/'R20_PREPARATION.json', dict(status='PREPARED_BUILD_NOT_EXECUTED',
        frozen_root=str(frozen), candidate_root=str(candidate), source_manifest=manifest,
        dependencies={rel: sha(frozen/rel) for rel in required}, frozen_binaries=binaries))
    print(candidate)


if __name__ == '__main__':
    main()
