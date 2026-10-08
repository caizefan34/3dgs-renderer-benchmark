import io
base='/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/rasterization/'
fwd_old='const uint cidx = bucket_offset * (config::block_size_blend * config::block_size_blend) + (j % config::block_size_blend) * config::block_size_blend + thread_rank;'
fwd_new='const uint cidx = (bucket_offset - 1) * (32 * config::block_size_blend) + (j % 32) * config::block_size_blend + thread_rank;'
bwd_old='const uint cidx = bucket_idx * (config::block_size_blend * config::block_size_blend) + lane_idx * config::block_size_blend + idx;'
bwd_new='const uint cidx = bucket_idx * (32 * config::block_size_blend) + lane_idx * config::block_size_blend + idx;'
for fn,old,new in [('include/kernels_forward.cuh',fwd_old,fwd_new),('include/kernels_backward.cuh',bwd_old,bwd_new)]:
    p=base+fn
    s=open(p).read()
    n=s.count(old)
    assert n==1, (fn,'count',n)
    s=s.replace(old,new)
    open(p,'w').write(s)
    print('PATCHED',fn,'occurrences',n)
print('DONE')
