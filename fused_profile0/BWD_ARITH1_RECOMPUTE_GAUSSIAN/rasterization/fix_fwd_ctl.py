#!/usr/bin/env python3
NL=chr(10)
base='/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/rasterization/'
p=base+'src/forward.cu'
s=open(p).read()
old='    cudaMemcpyToSymbol(faster_gs::rasterization::kernels::bwd1_gauss_cache, &bwd1_cache_ptr, sizeof(float*));'+NL+'    cudaMemcpyToSymbol(faster_gs::rasterization::kernels::bwd1_cache_enable, &bwd1_enable, sizeof(unsigned int));'
new='    faster_gs::rasterization::kernels::bwd1_gauss_cache_host = bwd1_cache_ptr;'+NL+'    faster_gs::rasterization::kernels::bwd1_cache_enable_host = bwd1_enable;'
n=s.count(old)
print('stale MemcpyToSymbol count:', n)
if n==1:
    s=s.replace(old,new); open(p,'w').write(s); print('removed stale MemcpyToSymbol, set host globals')
elif 'bwd1_gauss_cache_host = bwd1_cache_ptr' in s:
    print('already patched')
else:
    raise SystemExit('anchor not found')
# verify no cudaMemcpyToSymbol remains
print('remaining cudaMemcpyToSymbol:', s.count('cudaMemcpyToSymbol'))
print('host global set present:', 'bwd1_gauss_cache_host = bwd1_cache_ptr' in s)
