#!/usr/bin/env python3
"""Status: IMPLEMENTED; MX build NOT_EXECUTED. Build experimental csrc ONLY.

Uses the actual MX builder, isolated build dir, correct PyInit_csrc module
name, and checks reference hashes before/after. Does not rebuild std_ll.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import torch
from torch.utils.cpp_extension import load
from R20_GRADIENT_PARITY_TESTS import sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-root', required=True)
    a = p.parse_args(); candidate = Path(a.candidate_root).resolve()
    prep = json.loads((candidate/'R20_PREPARATION.json').read_text(encoding='utf-8'))
    frozen = Path(prep['frozen_root']).resolve()
    if candidate == frozen or frozen in candidate.parents:
        raise RuntimeError('candidate is not isolated')
    for rel, expected in prep['frozen_binaries'].items():
        if sha(frozen/rel) != expected: raise RuntimeError('reference identity changed: '+rel)
    if sha(candidate/'gsplat/csrc.so') != prep['frozen_binaries']['gsplat/csrc.so']:
        raise RuntimeError('candidate main extension is not the frozen reference')
    for rel, record in prep['source_manifest']['files'].items():
        if sha(candidate/rel) != record['patched_sha256']: raise RuntimeError('overlay changed: '+rel)
    path = candidate/'gsplat/experimental/render/kernels/cuda/build.py'
    spec = importlib.util.spec_from_file_location('r20_actual_mx_builder', path)
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    params = builder.get_build_parameters()
    params.name = 'csrc'
    build = candidate/'.r20_build'; build.mkdir(exist_ok=False)
    # Import path ends in csrc, so the binary must export PyInit_csrc.
    module = load(name='csrc', sources=params.sources, extra_cflags=params.extra_cflags,
        extra_cuda_cflags=params.extra_cuda_cflags, extra_include_paths=params.extra_include_paths,
        extra_ldflags=params.extra_ldflags, build_directory=str(build), verbose=True)
    if getattr(module, 'higs_true_absgrad_version', 0) != 1:
        raise RuntimeError('compiled extension lacks True AbsGrad capability')
    destination = candidate/'gsplat/experimental/render/kernels/csrc.so'
    if destination.exists(): raise RuntimeError('candidate binary already exists')
    shutil.copyfile(module.__file__, destination)
    for rel, expected in prep['frozen_binaries'].items():
        if sha(frozen/rel) != expected: raise RuntimeError('reference identity changed: '+rel)
    write_json(candidate/'R20_BUILD_IDENTITY.json', dict(status='BUILD_PASS_PARITY_NOT_EXECUTED',
        binary=str(destination), binary_sha256=sha(destination), torch=torch.__version__,
        cuda=torch.version.cuda, device=torch.cuda.get_device_name(), build_parameters=params.__dict__))
    print(destination)


if __name__ == '__main__':
    main()
