"""Compile/run unmodified extracted real kernel bodies on local CUDA.

Shared headers are from the local public gsplat checkout, NOT the frozen MX
build. This test covers pixel accumulation and ON/OFF invariance only.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import torch
from torch.utils.cpp_extension import include_paths, CUDA_HOME

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
out = HERE/'local_checks'; out.mkdir(exist_ok=True)
target = HERE/'source/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu'
native = REPO/'HIGS_SOURCE_HANDOFF_ACTUAL/source/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu'
h = target.read_text(encoding='utf-8'); h = h[h.index('template<uint32_t CDIM, bool ABSGRAD>'):h.index('// Background gradient')]
r = native.read_text(encoding='utf-8'); r = r[r.index('template<uint32_t CDIM, typename scalar_t>'):r.index('void launch_rasterize_to_pixels_3dgs_bwd_kernel(')]
prefix = '#include "Common.h"\n#include "Utils.cuh"\n#include "RasterizeToPixels3DGSDevice.cuh"\n#include <cooperative_groups/reduce.h>\n'
source = prefix + '\nnamespace higs_kernels { using namespace gsplat; namespace cg=cooperative_groups;\n'+h+'\n}\n'
source += '\nnamespace reference_kernels { using namespace gsplat; namespace cg=cooperative_groups;\n'+r+'\n}\n'
source += (HERE/'_local_blend_driver.cpp.in').read_text(encoding='utf-8')
cu=out/'extracted_blend_test.cu'; cu.write_text(source,encoding='utf-8')
exe=out/'extracted_blend_test.exe'
pkg=REPO/'public_baselines/gsplat/gsplat'
cmd=[str(Path(CUDA_HOME)/'bin/nvcc.exe'),str(cu),'-o',str(exe),'-std=c++20','-allow-unsupported-compiler','-arch=sm_120',
     '-Xcompiler','/Zc:preprocessor','-DWIN32_LEAN_AND_MEAN','--expt-relaxed-constexpr']
for path in [*include_paths(device_type='cuda'),pkg/'cuda/include',pkg/'cuda/csrc',
             REPO/'wv-hier-r1-mx/evidence/raw/evidence/gsplat_source_snapshot/cuda/csrc/third_party/glm']:
    cmd+=['-I',str(path)]
torch_lib = Path(torch.__file__).parent/'lib'
cmd += ['-L', str(torch_lib), '-lc10', '-lc10_cuda', '-ltorch_cpu', '-ltorch_cuda', '-Xcompiler', '/MD']
p=subprocess.run(cmd,capture_output=True,text=True,errors='replace')
(out/'blend_compile.log').write_text(p.stdout+p.stderr,encoding='utf-8')
if p.returncode:
    print((p.stdout+p.stderr)[-4000:]); raise SystemExit(p.returncode)
env = os.environ.copy(); env['PATH'] = str(torch_lib)+os.pathsep+env['PATH']
p=subprocess.run([str(exe)],capture_output=True,text=True,errors='replace',env=env)
(out/'blend_run.log').write_text(p.stdout+p.stderr,encoding='utf-8')
record=json.loads(p.stdout) if p.stdout.strip().startswith('{') else {'status':'FAIL','stderr':p.stderr}
record.update(device=torch.cuda.get_device_name(),exit_code=p.returncode,
              target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              native_source_sha256=hashlib.sha256(native.read_bytes()).hexdigest())
(out/'blend_check.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2));raise SystemExit(p.returncode)
