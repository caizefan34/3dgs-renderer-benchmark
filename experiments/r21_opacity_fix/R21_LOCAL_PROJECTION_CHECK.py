"""Status: IMPLEMENTED. Scoped local GPU test of the exact frozen projection kernel.
Extracts the kernel body verbatim, using retrieved frozen headers. This is NOT
the frozen Linux binary, renderer backward, tile coverage or training parity.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import torch
from torch.utils.cpp_extension import CUDA_HOME, include_paths
from R21_VISIBILITY_REGRESSION_TESTS import HERE, REPO, opacity_fixtures, parity


DRIVER = r'''
#include <fstream>
#include <vector>
#include <iostream>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { std::cerr<<cudaGetErrorString(e); return 2; } } while(0)
int main(int argc,char**argv) {
    if(argc!=3) return 3;
    std::ifstream in(argv[1],std::ios::binary);
    int cfg[5]; in.read((char*)cfg,sizeof(cfg));
    int n=cfg[0],c=cfg[1],w=cfg[2],h=cfg[3]; bool aware=cfg[4];
    std::vector<float*> buffers;
    for(int size : {n*3,n*4,n*3,n,c*16,c*9}) {
        std::vector<float> host(size); in.read((char*)host.data(),size*4);
        if(!in) return 4;
        float *dev; CK(cudaMalloc(&dev,size*4)); CK(cudaMemcpy(dev,host.data(),size*4,cudaMemcpyHostToDevice));
        buffers.push_back(dev);
    }
    int *r; float *m,*d,*q; CK(cudaMalloc(&r,c*n*2*4)); CK(cudaMalloc(&m,c*n*2*4));
    CK(cudaMalloc(&d,c*n*4)); CK(cudaMalloc(&q,c*n*3*4));
    gsplat::projection_ewa_3dgs_fused_fwd_kernel<float><<<(c*n+255)/256,256>>>(
        0,n,1,c,n,buffers[0],nullptr,buffers[1],buffers[2],aware?buffers[3]:nullptr,
        buffers[4],buffers[5],w,h,.1f,.01f,1e10f,0.f,gsplat::CameraModelType::PINHOLE,r,m,d,q,nullptr);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<int> result(c*n*2); CK(cudaMemcpy(result.data(),r,c*n*2*4,cudaMemcpyDeviceToHost));
    std::ofstream out(argv[2],std::ios::binary); out.write((char*)result.data(),result.size()*4);
    for(auto p:buffers) CK(cudaFree(p)); CK(cudaFree(r)); CK(cudaFree(m)); CK(cudaFree(d)); CK(cudaFree(q));
    return out?0:5;
}
'''


def main():
    out = HERE/'local_checks'; out.mkdir(exist_ok=True)
    frozen = HERE/'evidence/frozen/gsplat'
    path = frozen/'cuda/csrc/ProjectionEWA3DGSFused.cu'
    text = path.read_text(encoding='utf-8')
    kernel = text[text.index('template<typename scalar_t>'):text.index('void launch_projection_ewa_3dgs_fused_fwd_kernel(')]
    cu = out/'projection_check.cu'
    cu.write_text('#include "Utils.cuh"\nnamespace gsplat { namespace cg=cooperative_groups;\n'+kernel+'\n}\n'+DRIVER)
    exe = out/'projection_check.exe'
    glm = REPO/'wv-hier-r1-mx/evidence/raw/evidence/gsplat_source_snapshot/cuda/csrc/third_party/glm'
    cmd = [str(Path(CUDA_HOME)/'bin/nvcc.exe'), str(cu), '-o', str(exe), '-std=c++20',
        '-allow-unsupported-compiler', '-arch=sm_120', '-Xcompiler', '/Zc:preprocessor',
        '-DWIN32_LEAN_AND_MEAN', '--expt-relaxed-constexpr']
    for p in [*include_paths(device_type='cuda'), frozen/'cuda/include', glm]: cmd += ['-I', str(p)]
    lib = Path(torch.__file__).parent/'lib'
    cmd += ['-L', str(lib), '-lc10', '-lc10_cuda', '-ltorch_cpu', '-ltorch_cuda', '-Xcompiler', '/MD']
    p = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    (out/'compile.log').write_text(p.stdout+p.stderr, encoding='utf-8')
    if p.returncode:
        print((p.stdout+p.stderr)[-5000:]); return p.returncode
    env = os.environ.copy(); env['PATH'] = str(lib)+os.pathsep+env['PATH']
    def project(f, aware, rows=None):
        _, act = parity.activate(f, 'cpu')
        if rows is not None:
            act = {k:v[rows] for k,v in act.items()}
        n = len(act['means']); c = len(f['viewmats'])
        if n == 0: return torch.zeros(c, 0, 2, dtype=torch.int32)
        with (out/'input.bin').open('wb') as fp:
            np.array([n,c,f['width'],f['height'],int(aware)], dtype='<i4').tofile(fp)
            for t in (act['means'],act['quats'],act['scales'],act['opacities'],f['viewmats'],f['Ks']):
                t.detach().contiguous().numpy().astype('<f4').tofile(fp)
        subprocess.run([str(exe),str(out/'input.bin'),str(out/'output.bin')], check=True, env=env)
        return torch.from_numpy(np.fromfile(out/'output.bin',dtype='<i4').copy()).reshape(c,n,2)
    reports = {}; passed = True
    known = json.loads((HERE/'evidence/KNOWN_MISMATCH_FIXTURES.json').read_text())
    for name, f in opacity_fixtures().items():
        on, off = project(f, True), project(f, False)
        ids = (on>0).all(-1).any(0).nonzero().flatten()
        compact = project(f, True, ids)
        eq = torch.equal(compact, on[:,ids])
        subset = bool(((on>0).all(-1) <= (off>0).all(-1)).all())
        row = dict(master_compact_exact=eq, on_visibility_subset_off=subset,
                   on_off_radii_changed_components=int((on!=off).sum()))
        if name.startswith('known_'):
            rec = known[name[6:]]; expected = torch.tensor(rec['radii_std'],dtype=torch.int32)
            row['frozen_mx_radii_equal'] = torch.equal(on, expected)
            row['frozen_mx_visibility_equal'] = torch.equal((on>0).all(-1).any(0),torch.tensor(rec['std_visibility']))
            passed &= row['frozen_mx_radii_equal'] and row['frozen_mx_visibility_equal']
        passed &= eq and subset; reports[name] = row
    report = dict(status='PASS_SCOPED' if passed else 'FAIL', cases=reports,
        scope='Exact frozen projection kernel body + frozen Utils/Common headers; local RTX compilation; not full MX binary parity',
        frozen_kernel_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        extracted_kernel_sha256=hashlib.sha256(kernel.encode()).hexdigest(),
        device=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
        glm_source=str(glm), command=cmd, rgb_alpha_parity='NOT_EXECUTED', backward_parity='NOT_EXECUTED',
        mx_p4='NOT_EXECUTED', training='NOT_EXECUTED')
    (out/'PROJECTION_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2)); return 0 if passed else 1


if __name__ == '__main__': raise SystemExit(main())
