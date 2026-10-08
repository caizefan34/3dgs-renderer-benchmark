import sys
P = '/root/epic05_dpc0/src_full/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackwardDpc.cu'
s = open(P, encoding='utf-8').read()
orig = s
applied = [0]
def rep(old, new, n=1):
    global s
    c = s.count(old)
    if c != n:
        print('ANCHOR FAIL count=%d expected=%d for: %r' % (c, n, old[:70]))
        sys.exit(1)
    s = s.replace(old, new)
    applied[0] += 1
    print('OK[%d]: %r' % (applied[0], old[:55]))

NEW_KERNELS = r'''// ============================================================================
// DPC0 Stage 2 reduce: deterministic per-gaussian reduction of the projection
// VJP partials over cameras. One thread per visible gaussian g; fixed-order
// loop over camera index i (== ascending idx == i*N+g). Writes master
// means/quats/scales. No floating-point atomicAdd.
// ============================================================================
__global__ void dpc_reduce_proj_master_kernel(
    const uint32_t I,
    const uint32_t N,
    const float *__restrict__ part_means,   // [I*N, 3]
    const float *__restrict__ part_quats,   // [I*N, 4]
    const float *__restrict__ part_scales,  // [I*N, 3]
    const int64_t *__restrict__ visible_ids,// [N]
    float *__restrict__ grad_means,         // [M, 3]
    float *__restrict__ grad_quats,         // [M, 4]
    float *__restrict__ grad_scales         // [M, 3]
)
{
    const int64_t g = cg::this_grid().thread_rank();
    if(g >= N){ return; }
    const int64_t m_id = visible_ids[g];
    float mx=0.f,my=0.f,mz=0.f,q0=0.f,q1=0.f,q2=0.f,q3=0.f,s0=0.f,s1=0.f,s2=0.f;
    for(uint32_t i=0;i<I;++i){
        const int64_t idx=(int64_t)i*N+g;
        const float* pm=part_means+idx*3;
        mx+=pm[0]; my+=pm[1]; mz+=pm[2];
        const float* pq=part_quats+idx*4;
        q0+=pq[0]; q1+=pq[1]; q2+=pq[2]; q3+=pq[3];
        const float* ps=part_scales+idx*3;
        s0+=ps[0]; s1+=ps[1]; s2+=ps[2];
    }
    float* gm=grad_means+m_id*3;   gm[0]=mx; gm[1]=my; gm[2]=mz;
    float* gq=grad_quats+m_id*4;   gq[0]=q0; gq[1]=q1; gq[2]=q2; gq[3]=q3;
    float* gs=grad_scales+m_id*3;  gs[0]=s0; gs[1]=s1; gs[2]=s2;
}

// ============================================================================
// DPC0 Stage 3a reduce: deterministic per-gaussian reduction of the SH coeff
// partials over cameras -> master coeffs; and the SH-direction means partials
// reduced over cameras and ADDED onto the master means already written by
// dpc_reduce_proj_master_kernel. No floating-point atomicAdd.
// ============================================================================
__global__ void dpc_reduce_sh_master_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const float *__restrict__ part_coeffs,  // [I*N, K, D]
    const float *__restrict__ part_shmeans, // [I*N, 3]
    const int64_t *__restrict__ visible_ids,// [N]
    float *__restrict__ grad_colors,        // [M, K, D]
    float *__restrict__ grad_means          // [M, 3]
)
{
    const int64_t g = cg::this_grid().thread_rank();
    if(g >= N){ return; }
    const int64_t m_id = visible_ids[g];
    const int64_t KD = (int64_t)K*D;
    for(uint32_t k=0;k<K;++k){
        for(uint32_t d=0;d<D;++d){
            float acc=0.f;
            for(uint32_t i=0;i<I;++i){
                acc += part_coeffs[((int64_t)i*N+g)*KD + (int64_t)k*D + d];
            }
            grad_colors[m_id*KD + (int64_t)k*D + d] = acc;
        }
    }
    float mx=0.f,my=0.f,mz=0.f;
    for(uint32_t i=0;i<I;++i){
        const float* pm=part_shmeans+((int64_t)i*N+g)*3;
        mx+=pm[0]; my+=pm[1]; mz+=pm[2];
    }
    float* gm=grad_means+m_id*3;
    gm[0]+=mx; gm[1]+=my; gm[2]+=mz;
}
'''

A1o = '''higs_rasterize_backward_dpc(
    const at::Tensor &means2d,'''
A1n = NEW_KERNELS + A1o
rep(A1o, A1n)

A2o = '''    float *__restrict__ v_means,  // [M, 3]
    float *__restrict__ v_quats,  // [M, 4]
    float *__restrict__ v_scales  // [M, 3]
)'''
A2n = '''    float *__restrict__ v_means,  // [M, 3]
    float *__restrict__ v_quats,  // [M, 4]
    float *__restrict__ v_scales,  // [M, 3]
    float *__restrict__ part_means,   // [I*N, 3] DPC per-(i,g) partial
    float *__restrict__ part_quats,   // [I*N, 4]
    float *__restrict__ part_scales   // [I*N, 3]
)'''
rep(A2o, A2n)

A3o = '''    if(warp_group_g.thread_rank() == 0)
    {
        float *v_m = v_means + m_id * 3;
        gpuAtomicAdd(v_m, v_mean.x);
        gpuAtomicAdd(v_m + 1, v_mean.y);
        gpuAtomicAdd(v_m + 2, v_mean.z);
        float *v_q = v_quats + m_id * 4;
        gpuAtomicAdd(v_q, v_quat[0]);
        gpuAtomicAdd(v_q + 1, v_quat[1]);
        gpuAtomicAdd(v_q + 2, v_quat[2]);
        gpuAtomicAdd(v_q + 3, v_quat[3]);
        float *v_s = v_scales + m_id * 3;
        gpuAtomicAdd(v_s, v_scale[0]);
        gpuAtomicAdd(v_s + 1, v_scale[1]);
        gpuAtomicAdd(v_s + 2, v_scale[2]);
    }'''
A3n = '''    if(warp_group_g.thread_rank() == 0)
    {
        // DPC0: write per-(i,g) partial (idx unique -> no atomic).
        // Reduced deterministically over cameras by dpc_reduce_proj_master_kernel.
        float *v_m = part_means + idx * 3;
        v_m[0]=v_mean.x; v_m[1]=v_mean.y; v_m[2]=v_mean.z;
        float *v_q = part_quats + idx * 4;
        v_q[0]=v_quat[0]; v_q[1]=v_quat[1]; v_q[2]=v_quat[2]; v_q[3]=v_quat[3];
        float *v_s = part_scales + idx * 3;
        v_s[0]=v_scale[0]; v_s[1]=v_scale[1]; v_s[2]=v_scale[2];
    }'''
rep(A3o, A3n)

A4o = '''    float *__restrict__ v_coeffs,             // [M, K, D] master
    float *__restrict__ v_means               // [M, 3] master
)'''
A4n = '''    float *__restrict__ v_coeffs,             // [M, K, D] master
    float *__restrict__ v_means,              // [M, 3] master
    float *__restrict__ part_coeffs,          // [I*N, K, D] DPC per-(i,g) partial
    float *__restrict__ part_shmeans           // [I*N, 3] DPC per-(i,g) partial
)'''
rep(A4o, A4n)

A5o = '''        float *v_coeff_ptr        = v_coeffs + m_id * K * D;'''
A5n = '''        float *v_coeff_ptr        = part_coeffs + idx * K * D;'''
rep(A5o, A5n)

A6o = '''    if(!masked)
    {
        if(c == 0 && complete_group)
        {
            atomicAdd(v_means + m_id * 3 + 0, v_dir_sum.x);
            atomicAdd(v_means + m_id * 3 + 1, v_dir_sum.y);
            atomicAdd(v_means + m_id * 3 + 2, v_dir_sum.z);
        }
        else if(!complete_group)
        {
            atomicAdd(v_means + m_id * 3 + 0, v_dir.x);
            atomicAdd(v_means + m_id * 3 + 1, v_dir.y);
            atomicAdd(v_means + m_id * 3 + 2, v_dir.z);
        }
        // else: c != 0 lane of a complete group; its v_dir is already in the
        // leader's shuffled sum.
    }'''
A6n = '''    if(!masked)
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
    }'''
rep(A6o, A6n)

A7o = '''    at::Tensor v_opacities_flat = at::zeros({I * N}, opts);'''
A7n = '''    at::Tensor v_opacities_flat = at::zeros({I * N}, opts);
    at::Tensor part_means   = at::empty({I * N, 3}, opts);
    at::Tensor part_quats   = at::empty({I * N, 4}, opts);
    at::Tensor part_scales  = at::empty({I * N, 3}, opts);
    at::Tensor part_shmeans = at::empty({I * N, 3}, opts);'''
rep(A7o, A7n)

A8o = '''        const int64_t K = sh_coeffs.value().size(-2);
        const int64_t D = sh_coeffs.value().size(-1);'''
A8n = '''        const int64_t K = sh_coeffs.value().size(-2);
        const int64_t D = sh_coeffs.value().size(-1);
        at::Tensor part_coeffs = at::empty({I * N * K * D}, opts);'''
rep(A8o, A8n)

A9o = '''                visible_ids.const_data_ptr<int64_t>(),
                grad_means.data_ptr<float>(),
                grad_quats.data_ptr<float>(),
                grad_scales.data_ptr<float>()
            );'''
A9n = '''                visible_ids.const_data_ptr<int64_t>(),
                grad_means.data_ptr<float>(),
                grad_quats.data_ptr<float>(),
                grad_scales.data_ptr<float>(),
                part_means.data_ptr<float>(),
                part_quats.data_ptr<float>(),
                part_scales.data_ptr<float>()
            );'''
rep(A9o, A9n, n=2)

A10o = '''            visible_ids.const_data_ptr<int64_t>(),
            grad_colors.data_ptr<float>(),
            grad_means.data_ptr<float>()
        );'''
A10n = '''            visible_ids.const_data_ptr<int64_t>(),
            grad_colors.data_ptr<float>(),
            grad_means.data_ptr<float>(),
            part_coeffs.data_ptr<float>(),
            part_shmeans.data_ptr<float>()
        );'''
rep(A10o, A10n)

A11o = '''        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

        if(sh_coeffs.has_value() && color_dim >= 3)'''
A11n = '''        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- DPC0 Stage 2 reduce: deterministic per-gaussian reduction of the
    //      projection partials over cameras -> master means/quats/scales. ----
    {
        const uint32_t rthreads = 256;
        const uint32_t rgrid   = static_cast<uint32_t>((N + rthreads - 1) / rthreads);
        dpc_reduce_proj_master_kernel<<<rgrid, rthreads, 0, stream>>>(
            static_cast<uint32_t>(I), static_cast<uint32_t>(N),
            part_means.data_ptr<float>(), part_quats.data_ptr<float>(),
            part_scales.data_ptr<float>(), visible_ids.const_data_ptr<int64_t>(),
            grad_means.data_ptr<float>(), grad_quats.data_ptr<float>(),
            grad_scales.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

        if(sh_coeffs.has_value() && color_dim >= 3)'''
rep(A11o, A11n)

A12o = '''        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    else
    {
        // ---- Stage 3b: RGB mode - reduce evaluated-color grads over views ----'''
A12n = '''        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // ---- DPC0 Stage 3a reduce: deterministic per-gaussian reduction of
        //      the SH coeff partials over cameras -> master coeffs, and the
        //      SH-direction means partials added onto master means. ----
        {
            const uint32_t rthreads = 256;
            const uint32_t rgrid   = static_cast<uint32_t>((N + rthreads - 1) / rthreads);
            dpc_reduce_sh_master_kernel<<<rgrid, rthreads, 0, stream>>>(
                static_cast<uint32_t>(I), static_cast<uint32_t>(N),
                static_cast<uint32_t>(K), static_cast<uint32_t>(D),
                part_coeffs.data_ptr<float>(), part_shmeans.data_ptr<float>(),
                visible_ids.const_data_ptr<int64_t>(),
                grad_colors.data_ptr<float>(), grad_means.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    else
    {
        // ---- Stage 3b: RGB mode - reduce evaluated-color grads over views ----'''
rep(A12o, A12n)

if s == orig:
    print('NO CHANGES'); sys.exit(1)
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED OK replacements=%d' % applied[0])
