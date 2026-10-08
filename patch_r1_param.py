#!/usr/bin/env python3
"""Idempotent patcher: forward.cu + backward.cu to use kernel-param cache (host globals).
Kernels + symbol files already patched. This script patches forward.cu and backward.cu
and is safe to re-run (checks for existing patch)."""
base='/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/rasterization/'

# ---- forward.cu ----
p=base+'src/forward.cu'
s=open(p).read()
# ctl block: replace cudaMemcpyToSymbol with host globals
ctl_old='''    cudaMemcpyToSymbol(faster_gs::rasterization::kernels::bwd1_gauss_cache, &bwd1_cache_ptr, sizeof(float*));
    cudaMemcpyToSymbol(faster_gs::rasterization::kernels::bwd1_cache_enable, &bwd1_enable, sizeof(unsigned int));'''
ctl_new='''    faster_gs::rasterization::kernels::bwd1_gauss_cache_host = bwd1_cache_ptr;
    faster_gs::rasterization::kernels::bwd1_cache_enable_host = bwd1_enable;'''
if ctl_old in s:
    s=s.replace(ctl_old,ctl_new); print('fwd ctl patched')
elif ctl_new in s:
    print('fwd ctl already patched')
else:
    raise SystemExit('fwd ctl anchor NOT found')
# launch: add host globals after grid.x
lnch_old='        grid.x\n    );\n    CHECK_CUDA(config::debug, "blend");'
lnch_new='''        grid.x,
        faster_gs::rasterization::kernels::bwd1_gauss_cache_host,
        faster_gs::rasterization::kernels::bwd1_cache_enable_host
    );
    CHECK_CUDA(config::debug, "blend");'''
if lnch_old in s:
    s=s.replace(lnch_old,lnch_new); print('fwd launch patched')
elif 'bwd1_gauss_cache_host' in s.split('blend_cu')[1][:800]:
    print('fwd launch already patched')
else:
    raise SystemExit('fwd launch anchor NOT found')
open(p,'w').write(s)

# ---- backward.cu ----
p=base+'src/backward.cu'
s=open(p).read()
if '#include "bwd1_cache_symbol.h"' not in s:
    inc_old='#include "helper_math.h"\n#include <stdexcept>'
    assert inc_old in s, 'bwd inc anchor NOT found'
    s=s.replace(inc_old, inc_old+'\n#include "bwd1_cache_symbol.h"',1)
    print('bwd include patched')
else:
    print('bwd include already patched')
lnch_old='        grid.x\n    );\n    CHECK_CUDA(config::debug, "blend_backward");'
lnch_new='''        grid.x,
        faster_gs::rasterization::kernels::bwd1_gauss_cache_host,
        faster_gs::rasterization::kernels::bwd1_cache_enable_host
    );
    CHECK_CUDA(config::debug, "blend_backward");'''
if lnch_old in s:
    s=s.replace(lnch_old,lnch_new); print('bwd launch patched')
elif 'bwd1_gauss_cache_host' in s:
    print('bwd launch already patched')
else:
    raise SystemExit('bwd launch anchor NOT found')
open(p,'w').write(s)
print('DONE')
