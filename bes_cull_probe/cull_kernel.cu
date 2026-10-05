// cull_kernel.cu - gsplat-style forward alpha-compositing walk with per-tile
// SHARED-MEMORY staging and EXACT no-op culling.
//
// One block per 16x16 tile, one thread per pixel. The tile's depth-sorted
// gaussian list is staged into shared memory in chunks (cooperative load), so
// each gaussian's params are read from global memory ONCE per tile (not once
// per pixel). All 256 pixels then walk the chunk from shared memory.
//
// Modes:
//   0 baseline : every occurrence pays q + exp + threshold + blend
//   1 culled   : skip exp + blend when q > gthr[gid] (provably alpha<cutoff)
//   2 floor    : stage + loop only, no q/exp/blend (memory/loop lower bound)
//
// Culling test: alpha<cutoff <=> q > 2*(ln(op)+ln(255)) = gthr. Algebraically
// identical to the baseline threshold, so modes 0 and 1 are bit-identical.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TS 16
#define CHUNK 512          // gaussians staged per chunk (10*512*4 = 20KB smem)
#define ALPHA_CAP 0.999f
#define CULL_MARGIN 0.01f   // numerical safeguard: skip only when alpha provably < cutoff
#define NARG 10

__global__ void walk_kernel(
    const float* __restrict__ gmx, const float* __restrict__ gmy,
    const float* __restrict__ gca, const float* __restrict__ gcb,
    const float* __restrict__ gcc, const float* __restrict__ gop,
    const float* __restrict__ gthr, const float* __restrict__ gcol,
    const int*   __restrict__ gids, const int*   __restrict__ tile_ptr,
    int num_tiles_x, int num_tiles_y, int W, int H, int mode,
    float* __restrict__ out_img)
{
    extern __shared__ float smem[];
    float* s_mx = smem;
    float* s_my = s_mx + CHUNK;
    float* s_ca = s_my + CHUNK;
    float* s_cb = s_ca + CHUNK;
    float* s_cc = s_cb + CHUNK;
    float* s_op = s_cc + CHUNK;
    float* s_thr = s_op + CHUNK;
    float* s_cr = s_thr + CHUNK;
    float* s_cg = s_cr + CHUNK;
    float* s_cb2 = s_cg + CHUNK;

    int tile = blockIdx.x;
    int tx = tile % num_tiles_x;
    int ty = tile / num_tiles_x;
    int px = tx * TS + threadIdx.x;
    int py = ty * TS + threadIdx.y;
    bool inside = (px < W && py < H);
    int begin = tile_ptr[tile];
    int end = tile_ptr[tile + 1];
    int L = end - begin;
    float fx = (float)px, fy = (float)py;
    float Cr = 0.0f, Cg = 0.0f, Cb = 0.0f, T = 1.0f;
    int li = threadIdx.y * TS + threadIdx.x;   // 0..255

    for (int chunk = 0; chunk < L; chunk += CHUNK) {
        int cl = min(CHUNK, L - chunk);
        for (int s = li; s < cl; s += TS * TS) {   // grid-stride: stage all cl entries
            int gid = gids[begin + chunk + s];
            s_mx[s] = gmx[gid]; s_my[s] = gmy[gid];
            s_ca[s] = gca[gid]; s_cb[s] = gcb[gid]; s_cc[s] = gcc[gid];
            s_op[s] = gop[gid]; s_thr[s] = gthr[gid];
            s_cr[s] = gcol[gid * 3]; s_cg[s] = gcol[gid * 3 + 1];
            s_cb2[s] = gcol[gid * 3 + 2];
        }
        __syncthreads();
        if (inside && mode != 2) {
            for (int j = 0; j < cl; ++j) {
                float dx = fx - s_mx[j];
                float dy = fy - s_my[j];
                float q = s_ca[j]*dx*dx + 2.0f*s_cb[j]*dx*dy + s_cc[j]*dy*dy;
                float op = s_op[j];
                float alpha;
                if (mode == 1) {
                    if (q > s_thr[j] + CULL_MARGIN) { alpha = 0.0f; }
                    else {
                        alpha = op * expf(-0.5f * q);
                        if (alpha < (1.0f/255.0f)) alpha = 0.0f;
                        else if (alpha > ALPHA_CAP) alpha = ALPHA_CAP;
                    }
                } else {
                    alpha = op * expf(-0.5f * q);
                    if (alpha < (1.0f/255.0f)) alpha = 0.0f;
                    else if (alpha > ALPHA_CAP) alpha = ALPHA_CAP;
                }
                if (alpha > 0.0f) {
                    float w = T * alpha;
                    Cr += w * s_cr[j]; Cg += w * s_cg[j]; Cb += w * s_cb2[j];
                    T *= (1.0f - alpha);
                }
            }
        }
        __syncthreads();
    }
    if (inside) {
        int p = py * W + px;
        out_img[p * 3 + 0] = Cr;
        out_img[p * 3 + 1] = Cg;
        out_img[p * 3 + 2] = Cb;
    }
}

std::vector<torch::Tensor> walk(
    torch::Tensor gmx, torch::Tensor gmy, torch::Tensor gca, torch::Tensor gcb,
    torch::Tensor gcc, torch::Tensor gop, torch::Tensor gthr, torch::Tensor gcol,
    torch::Tensor gids, torch::Tensor tile_ptr,
    int num_tiles_x, int num_tiles_y, int W, int H, int mode)
{
    int n_pix = W * H;
    auto out = torch::zeros({n_pix, 3}, gmx.options());   // zero-init: border=0
    int num_tiles = num_tiles_x * num_tiles_y;
    dim3 block(TS, TS);
    dim3 grid(num_tiles);
    size_t shmem = (size_t)NARG * CHUNK * sizeof(float);
    walk_kernel<<<grid, block, shmem>>>(
        gmx.data_ptr<float>(), gmy.data_ptr<float>(),
        gca.data_ptr<float>(), gcb.data_ptr<float>(), gcc.data_ptr<float>(),
        gop.data_ptr<float>(), gthr.data_ptr<float>(), gcol.data_ptr<float>(),
        gids.data_ptr<int>(), tile_ptr.data_ptr<int>(),
        num_tiles_x, num_tiles_y, W, H, mode, out.data_ptr<float>());
    return {out};
}
