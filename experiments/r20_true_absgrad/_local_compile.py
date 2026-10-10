"""Compile the actual overlay translation unit against available local headers.

This is a compile-only check, NOT frozen-MX numerical parity evidence.
"""
from pathlib import Path
import json
import subprocess
import sys
import torch
from torch.utils.cpp_extension import include_paths, CUDA_HOME
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
pkg = REPO / 'public_baselines/gsplat/gsplat'
src = HERE / 'source/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu'
out = HERE / 'local_checks'
out.mkdir(exist_ok=True)
cmd = [str(Path(CUDA_HOME) / 'bin/nvcc.exe'), '-c', str(src), '-o', str(out / 'HigsNativeBackward.obj'),
       '-std=c++20', '-allow-unsupported-compiler', '-arch=sm_120', '-Xcompiler', '/Zc:preprocessor',
       '-DWIN32_LEAN_AND_MEAN', '--expt-relaxed-constexpr']
for path in [*include_paths(device_type='cuda'), pkg/'cuda/include', pkg/'cuda/csrc',
             REPO/'wv-hier-r1-mx/evidence/raw/evidence/gsplat_source_snapshot/cuda/csrc/third_party/glm']:
    cmd += ['-I', str(path)]
p = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
(out / 'compile.log').write_text(p.stdout + p.stderr, encoding='utf-8')
record = {'status': 'PASS' if p.returncode == 0 else 'FAIL', 'exit_code': p.returncode,
          'scope': 'LOCAL_COMPILE_ONLY_PUBLIC_HEADERS_NOT_MX_BINARY_PARITY',
          'torch': torch.__version__, 'cuda': torch.version.cuda, 'command': cmd}
(out / 'compile.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record, indent=2))
print((p.stdout+p.stderr)[-7000:])
if p.returncode == 0:
    ext = HERE/'source/gsplat/experimental/render/kernels/cuda/ext.cpp'
    cmd = ['cl.exe', '/c', str(ext), '/Fo'+str(out/'ext.obj'), '/std:c++20', '/Zc:preprocessor',
           '/DWIN32_LEAN_AND_MEAN', '/EHsc', '/MD']
    for path in [*include_paths(device_type='cuda'), Path(sys.prefix)/'include', pkg/'cuda/include', pkg/'cuda/csrc',
                 REPO/'HIGS_SOURCE_HANDOFF_ACTUAL/source/gsplat/experimental/render/kernels/cuda',
                 REPO/'wv-hier-r1-mx/evidence/raw/evidence/gsplat_source_snapshot/cuda/csrc/third_party/glm']:
        cmd += ['/I'+str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    (out/'binding_compile.log').write_text(p.stdout+p.stderr, encoding='utf-8')
    (out/'binding_compile.json').write_text(json.dumps({'status': 'PASS' if p.returncode == 0 else 'FAIL',
        'exit_code': p.returncode, 'scope': 'LOCAL_BINDING_COMPILE_ONLY', 'command': cmd}, indent=2), encoding='utf-8')
    print('binding compile exit:', p.returncode)
    print((p.stdout+p.stderr)[-2500:])
raise SystemExit(p.returncode)
