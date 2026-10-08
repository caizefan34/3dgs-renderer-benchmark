#!/usr/bin/env python3
NL=chr(10)
base='/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/rasterization/'
G='faster_gs::rasterization::kernels::bwd1_gauss_cache_host'
E='faster_gs::rasterization::kernels::bwd1_cache_enable_host'
# forward.cu launch
p=base+'src/forward.cu'
s=open(p).read()
anchor='        grid.x'+NL+'    );'+NL+'    CHECK_CUDA(config::debug, "blend");'
repl='        grid.x,'+NL+'        '+G+','+NL+'        '+E+NL+'    );'+NL+'    CHECK_CUDA(config::debug, "blend");'
n=s.count(anchor)
print('fwd anchor count:', n)
if n==1:
    s=s.replace(anchor,repl); open(p,'w').write(s); print('fwd launch patched')
elif G in s:
    print('fwd already patched')
else:
    raise SystemExit('fwd anchor fail')
# backward.cu
p=base+'src/backward.cu'
s=open(p).read()
if '#include "bwd1_cache_symbol.h"' not in s:
    s=s.replace('#include "helper_math.h"'+NL+'#include <stdexcept>','#include "helper_math.h"'+NL+'#include <stdexcept>'+NL+'#include "bwd1_cache_symbol.h"',1)
    print('bwd include patched')
anchor2='        grid.x'+NL+'    );'+NL+'    CHECK_CUDA(config::debug, "blend_backward");'
repl2='        grid.x,'+NL+'        '+G+','+NL+'        '+E+NL+'    );'+NL+'    CHECK_CUDA(config::debug, "blend_backward");'
n2=s.count(anchor2)
print('bwd anchor count:', n2)
if n2==1:
    s=s.replace(anchor2,repl2); open(p,'w').write(s); print('bwd launch patched')
elif G in s:
    print('bwd already patched')
else:
    raise SystemExit('bwd anchor fail')
print('DONE')
