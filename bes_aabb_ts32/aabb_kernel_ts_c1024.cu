// aabb_kernel.cu - gsplat-style forward alpha-compositing walk with PER-TILE
// EXACT AABB REJECT (mode 3), PER-TILE+PER-PIXEL (mode 4), and a skip
// DIAGNOSTIC (mode 5: writes per-pixel max alpha among SKIPPED gaussians).
//
// Modes:
//   0 baseline : every occurrence pays q + exp + threshold + blend
//   3 tile     : skip whole (tile,gaussian) if its AABB misses the tile
//   4 tile+pix : mode 3 + per-pixel |dx|>dx_max || |dy|>dy_max reject before q
//   5 diag     : mode 3 skip, but record per-pixel max alpha among skipped (mode-0 formula)
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>


#define CHUNK 1024
#define ALPHA_CAP 0.999f
#define CULL_MARGIN 0.01f
#define NARG 13

__global__ void walk_kernel(
    const float* __restrict__ gmx, const float* __restrict__ gmy,
    const float* __restrict__ gca, const float* __restrict__ gcb,
    const float* __restrict__ gcc, const float* __restrict__ gop,
    const float* __restrict__ gthr, const float* __restrict__ gcol,
    const float* __restrict__ gdx, const float* __restrict__ gdy,
    const int*   __restrict__ gids, const int* __restrict__ tile_ptr,
    int num_tiles_x, int num_tiles_y, int W, int H, int mode, int TS,
    float* __restrict__ out_img, int* __restrict__ out_gid)
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
    float* s_dx = s_cb2 + CHUNK;
    float* s_dy = s_dx + CHUNK;
    float* s_keep = s_dy + CHUNK;

    int tile = blockIdx.x;
    int tx = tile % num_tiles_x;
    int ty = tile / num_tiles_x;
    int px = tx * TS + threadIdx.x;
    int py = ty * TS + threadIdx.y;
    bool inside = (px < W && py < H);
    float tile_x0 = (float)(tx * TS);
    float tile_x1 = (float)(tx * TS + TS - 1);
    float tile_y0 = (float)(ty * TS);
    float tile_y1 = (float)(ty * TS + TS - 1);
    int begin = tile_ptr[tile];
    int end = tile_ptr[tile + 1];
    int L = end - begin;
    float fx = (float)px, fy = (float)py;
    float Cr = 0.0f, Cg = 0.0f, Cb = 0.0f, T = 1.0f;
    float skipmax = 0.0f;
    int skipgid = -1;
    int li = threadIdx.y * TS + threadIdx.x;
    bool do_reject = (mode == 3 || mode == 4 || mode == 5);

    for (int chunk = 0; chunk < L; chunk += CHUNK) {
        int cl = min(CHUNK, L - chunk);
        for (int s = li; s < cl; s += TS * TS) {
            int gid = gids[begin + chunk + s];
            float mx = gmx[gid], my = gmy[gid];
            float dxm = gdx[gid], dym = gdy[gid];
            s_mx[s] = mx; s_my[s] = my;
            s_ca[s] = gca[gid]; s_cb[s] = gcb[gid]; s_cc[s] = gcc[gid];
            s_op[s] = gop[gid]; s_thr[s] = gthr[gid];
            s_cr[s] = gcol[gid * 3]; s_cg[s] = gcol[gid * 3 + 1];
            s_cb2[s] = gcol[gid * 3 + 2];
            s_dx[s] = dxm; s_dy[s] = dym;
            if (do_reject) {
                s_keep[s] = (mx + dxm >= tile_x0 && mx - dxm <= tile_x1 &&
                             my + dym >= tile_y0 && my - dym <= tile_y1) ? 1.0f : 0.0f;
            }
        }
        __syncthreads();
        if (inside && mode != 2) {
            for (int j = 0; j < cl; ++j) {
                bool skipped = (do_reject && s_keep[j] < 0.5f);
                if (skipped) {
                    if (mode == 5) {
                        float dx = fx - s_mx[j];
                        float dy = fy - s_my[j];
                        float q = s_ca[j]*dx*dx + 2.0f*s_cb[j]*dx*dy + s_cc[j]*dy*dy;
                        float op = s_op[j];
                        float alpha;
                        if (q > s_thr[j] + CULL_MARGIN) { alpha = 0.0f; }
                        else {
                            alpha = op * expf(-0.5f * q);
                            if (alpha < (1.0f/255.0f)) alpha = 0.0f;
                            else if (alpha > ALPHA_CAP) alpha = ALPHA_CAP;
                        }
                        if (alpha > skipmax) { skipmax = alpha; skipgid = gids[begin + chunk + j]; }
                    }
                    continue;
                }
                if (mode == 4) {
                    float dx = fx - s_mx[j];
                    float dy = fy - s_my[j];
                    float adx = dx < 0 ? -dx : dx;
                    float ady = dy < 0 ? -dy : dy;
                    if (adx > s_dx[j] || ady > s_dy[j]) continue;
                }
                float dx = fx - s_mx[j];
                float dy = fy - s_my[j];
                float q = s_ca[j]*dx*dx + 2.0f*s_cb[j]*dx*dy + s_cc[j]*dy*dy;
                float op = s_op[j];
                float alpha;
                if (q > s_thr[j] + CULL_MARGIN) { alpha = 0.0f; }
                else {
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
        if (mode == 5) { out_img[p] = skipmax; out_gid[p] = skipgid; }
        else { out_img[p * 3 + 0] = Cr; out_img[p * 3 + 1] = Cg; out_img[p * 3 + 2] = Cb; }
    }
}

std::vector<torch::Tensor> walk(
    torch::Tensor gmx, torch::Tensor gmy, torch::Tensor gca, torch::Tensor gcb,
    torch::Tensor gcc, torch::Tensor gop, torch::Tensor gthr, torch::Tensor gcol,
    torch::Tensor gdx, torch::Tensor gdy,
    torch::Tensor gids, torch::Tensor tile_ptr,
    int num_tiles_x, int num_tiles_y, int W, int H, int mode, int TS)
{
    int n_pix = W * H;
    auto out = (mode == 5) ? torch::zeros({n_pix}, gmx.options())
                           : torch::zeros({n_pix, 3}, gmx.options());
    auto out_gid = torch::zeros({n_pix}, gmx.options().dtype(torch::kInt32));
    int num_tiles = num_tiles_x * num_tiles_y;
    dim3 block(TS, TS);
    dim3 grid(num_tiles);
    size_t shmem = (size_t)NARG * CHUNK * sizeof(float);
    cudaFuncSetAttribute(walk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    walk_kernel<<<grid, block, shmem>>>(
        gmx.data_ptr<float>(), gmy.data_ptr<float>(),
        gca.data_ptr<float>(), gcb.data_ptr<float>(), gcc.data_ptr<float>(),
        gop.data_ptr<float>(), gthr.data_ptr<float>(), gcol.data_ptr<float>(),
        gdx.data_ptr<float>(), gdy.data_ptr<float>(),
        gids.data_ptr<int>(), tile_ptr.data_ptr<int>(),
        num_tiles_x, num_tiles_y, W, H, mode, TS, out.data_ptr<float>(),
        out_gid.data_ptr<int>());
    return {out, out_gid};
}
