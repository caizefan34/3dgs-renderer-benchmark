import io
path = "/root/epic05_dpc0/src_full/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackwardDpc.cu"
with open(path, "r") as f:
    s = f.read()

old1 = """    // dir = means - cam_pos -> d(loss)/d(means) += v_dir. The D channel lanes
    // of one (camera, gaussian) are consecutive (c == t % D), so the three
    // partials reduce with two shuffle-downs and one atomicAdd per output
    // coordinate instead of three serialized same-address atomicAdds. Partial
    // warps (grid tail) and groups that straddle a warp boundary fall back to
    // the per-lane atomics; masked lanes contribute v_dir = 0 to the shuffle.
    const uint32_t lane       = static_cast<uint32_t>(t) & 31u;
    const uint32_t warp_start = static_cast<uint32_t>(t) & ~31u;
    const bool full_warp      = (static_cast<int64_t>(warp_start) + 32 <= num);
    const int32_t group_lane  = static_cast<int32_t>(lane) - static_cast<int32_t>(c);
    const bool complete_group = full_warp && group_lane >= 0 && (group_lane + 2 <= 31);

    vec3 v_dir_sum = v_dir;
    if(full_warp)
    {
        v_dir_sum.x += __shfl_down_sync(0xffffffffu, v_dir.x, 1);
        v_dir_sum.y += __shfl_down_sync(0xffffffffu, v_dir.y, 1);
        v_dir_sum.z += __shfl_down_sync(0xffffffffu, v_dir.z, 1);
        v_dir_sum.x += __shfl_down_sync(0xffffffffu, v_dir.x, 2);
        v_dir_sum.y += __shfl_down_sync(0xffffffffu, v_dir.y, 2);
        v_dir_sum.z += __shfl_down_sync(0xffffffffu, v_dir.z, 2);
    }

    if(!masked)
    {
        if(c == 0 && complete_group)
        {
            // DPC0: write per-(i,g) partial (idx unique -> no atomic).
            float *vm = part_shmeans + idx * 3;
            vm[0]=v_dir_sum.x; vm[1]=v_dir_sum.y; vm[2]=v_dir_sum.z;
        }
        else if(!complete_group)
        {
            // Fallback (partial-warp tail / group straddle): rare for the
            // frozen cells (D==3, N large); not hit for room/bicycle 10K/15K,
            // so run-to-run determinism is preserved for the gate cells.
            atomicAdd(part_shmeans + idx * 3 + 0, v_dir.x);
            atomicAdd(part_shmeans + idx * 3 + 1, v_dir.y);
            atomicAdd(part_shmeans + idx * 3 + 2, v_dir.z);
        }
        // else: c != 0 lane of a complete group; its v_dir is already in the
        // leader's shuffled sum.
    }
}"""

new1 = """    // dir = means - cam_pos -> d(loss)/d(means) += v_dir, summed over the D SH
    // channels of this (camera, gaussian). DPC0: each (i,g,c) thread writes its
    // OWN v_dir into a unique slot of part_shmeans[I*N, D, 3] (no shuffle, no
    // atomicAdd), so the partial is independent of warp grouping. The reduce
    // kernel sums the D channel slots per (i,g) then the I cameras in a fixed
    // order -> bit-identical across runs. Masked (invisible) lanes keep
    // v_dir == 0 and write zeros.
    float *vm = part_shmeans + (idx * D + c) * 3;
    vm[0] = v_dir.x;
    vm[1] = v_dir.y;
    vm[2] = v_dir.z;
}"""

assert old1 in s, "old1 not found"
s = s.replace(old1, new1, 1)

oldc = "    float *__restrict__ part_shmeans           // [I*N, 3] DPC per-(i,g) partial"
newc = "    float *__restrict__ part_shmeans           // [I*N, D, 3] DPC per-(i,g,c) partial"
assert oldc in s, "comment not found"
s = s.replace(oldc, newc, 1)

old2 = """    float mx=0.f,my=0.f,mz=0.f;
    for(uint32_t i=0;i<I;++i){
        const float* pm=part_shmeans+((int64_t)i*N+g)*3;
        mx+=pm[0]; my+=pm[1]; mz+=pm[2];
    }"""
new2 = """    float mx=0.f,my=0.f,mz=0.f;
    for(uint32_t i=0;i<I;++i){
        for(uint32_t c=0;c<D;++c){
            const float* pm=part_shmeans+(((int64_t)i*N+g)*D+c)*3;
            mx+=pm[0]; my+=pm[1]; mz+=pm[2];
        }
    }"""
assert old2 in s, "old2 not found"
s = s.replace(old2, new2, 1)

old3 = "    at::Tensor part_shmeans = at::empty({I * N, 3}, opts);"
new3 = "    at::Tensor part_shmeans = at::empty({I * N * D, 3}, opts);"
assert old3 in s, "old3 not found"
s = s.replace(old3, new3, 1)

with open(path, "w") as f:
    f.write(s)
print("OK all edits applied")