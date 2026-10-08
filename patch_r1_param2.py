#!/usr/bin/env python3
base='/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/rasterization/'
# forward.cu launch
p=base+'src/forward.cu'
s=open(p).read()
anchor='        grid.x\n    );\n    CHECK_CUDA(config::debug, "blend");'
repl='        grid.x,\n        faster_gs::rasterization::kernels::bwd1_gauss_cache_host,\n        faster_gs::rasterization::kernels::bwd1_cache_enable_host\n    );\n    CHECK_CUDA(config::debug, "blend");'
n=s.count(anchor)
print('fwd anchor count:', n)
if n==1:
    s=s.replace(anchor,repl); open(p,'w').write(s); print('fwd launch patched')
elif 'bwd1_gauss_cache_host' in s:
    print('fwd already patched')
else:
    raise SystemExit('fwd anchor fail')
# backward.cu
p=base+'src/backward.cu'
s=open(p).read()
if '#include "bwd1_cache_symbol.h"' not in s:
    s=s.replace('#include "helper_math.h"\n#include <stdexcept>','#include "helper_math.h"\n#include <stdexcept>\n#include "bwd1_cache_symbol.h"',1)
    print('bwd include patched')
anchor2='        grid.x\n    );\n    CHECK_CUDA(config::debug, "blend_backward");'
repl2='        grid.x,\n        faster_gs::rasterization::kernels::bwd1_gauss_cache_host,\n        faster_gs::rasterization::kernels::bwd1_cache_enable_host\n    );\n    CHECK_CUDA(config::debug, "blend_backward");'
n2=s.count(anchor2)
print('bwd anchor count:', n2)
if n2==1:
    s=s.replace(anchor2,repl2); open(p,'w').write(s); print('bwd launch patched')
elif 'bwd1_gauss_cache_host' in s:
    print('bwd already patched')
else:
    raise SystemExit('bwd anchor fail')
print('DONE')
