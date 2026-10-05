// aabb_kernel2.cu - TWO-STAGE gsplat forward walk.
// Stage 1 loads only the 4 AABB-test floats (mx,my,gdx,gdy) for every
// (tile,gaussian) pair and computes the per-tile keep flag. Stage 2 loads the
// remaining 9 floats (conic, opacity, threshold, color) ONLY for survivors and
// runs q + exp + threshold + blend. This cuts global-memory traffic for the
// ~82% of pairs rejected by the AABB from 13 floats to 4, attacking the
// staging floor that caps the single-stage per-tile AABB reject.
//
// Modes:
//   0 baseline : single-stage, every occurrence pays 13-float load + q + blend
//   6 tile     : two-stage, per-tile AABB reject (exact, bit-identical to 0)
//   7 tile+pix : two-stage, per-tile AABB + per-pixel |dx|>gdx || |dy|>gdy reject
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TS 16
#define CHUNK 512
#define ALPHA_CAP 0.999f
#define NARG 5   // s_mx,s_my,s_dx,s_dy,s_keep

__global__ void walk_kernel2(
    const float* __restrict__ gmx, const float* __restrict__ gmy,
    const float* __restrict__ gca, const float* __restrict__ gcb,
    const float* __restrict__ gcc, const float* __restrict__ gop,
    const float* __restrict__ gthr, const float* __restrict__ gcol,
    const float* __restrict__ gdx, const float* __restrict__ gdy,
    const int*   __restrict__ gids, const int*   __restrict__ tile_ptr,
    int num_tiles_x, int W, int H, int mode,
    float* __restrict__ out_img)
{
    extern __shared__ float smem[];
    float* s_mx = smem;
    float* s_my = s_mx + CHUNK;
    float* s_dx = s_my + CHUNK;
    float* s_dy = s_dx + CHUNK;
    float* s_keep = s_dy + CHUNK;

    int tile = blockIdx.x;
    int tx = tile % num_tiles_x;
    int px = tx * TS + threadIdx.x;
    int py = (tile / num_tiles_x) * TS + threadIdx.y;
    bool inside = (px < W && py < H);
    float tile_x0 = (float)(tx * TS);
    float tile_x1 = (float)(tx * TS + TS - 1);
    float tile_y0 = (float)((tile / num_tiles_x) * TS);
    float tile_y1 = (float)((tile / num_tiles_x) * TS + TS - 1);
    int begin = tile_ptr[tile];
    int end = tile_ptr[tile + 1];
    int L = end - begin;
    float fx = (float)px, fy = (float)py;
    float Cr = 0.0f, Cg = 0.0f, Cb = 0.0f, T = 1.0f;
    bool two_stage = (mode == 6 || mode == 7);

    for (int chunk = 0; chunk < L; chunk += CHUNK) {
        int cl = min(CHUNK, L - chunk);
        // ---- Stage 1: load 4 AABB floats + keep flag ----
        for (int s = threadIdx.y * TS + threadIdx.x; s < cl; s += TS * TS) {
            int gid = gids[begin + chunk + s];
            float mx = gmx[gid], my = gmy[gid];
            float dxm = gdx[gid], dym = gdy[gid];
            s_mx[s] = mx; s_my[s] = my; s_dx[s] = dxm; s_dy[s] = dym;
            if (two_stage) {
                s_keep[s] = (mx + dxm >= tile_x0 && mx - dxm <= tile_x1 &&
                             my + dym >= tile_y0 && my - dym <= tile_y1) ? 1.0f : 0.0f;
            }
        }
        __syncthreads();
        if (inside && mode != 8) {
            if (!two_stage) {
                // baseline single-stage: load all 13, no reject
                for (int j = 0; j < cl; ++j) {
                    int gid = gids[begin + chunk + j];
                    float dx = fx - gmx[gid];
                    float dy = fy - gmy[gid];
                    float q = gca[gid]*dx*dx + 2.0f*gcb[gid]*dx*dy + gcc[gid]*dy*dy;
                    float op = gop[gid];
                    float alpha;
                    if (q > gthr[gid] + 0.01f) { alpha = 0.0f; }
                    else {
                        alpha = op * expf(-0.5f * q);
                        if (alpha < (1.0f/255.0f)) alpha = 0.0f;
                        else if (alpha > ALPHA_CAP) alpha = ALPHA_CAP;
                    }
                    if (alpha > 0.0f) {
                        float w = T * alpha;
                        Cr += w * gcol[gid*3]; Cg += w * gcol[gid*3+1]; Cb += w * gcol[gid*3+2];
                        T *= (1.0f - alpha);
                    }
                }
            } else {
                // two-stage: only survivors pay the 9-float load + q + blend
                for (int j = 0; j < cl; ++j) {
                    if (s_keep[j] < 0.5f) continue;
                    int gid = gids[begin + chunk + j];
                    if (mode == 7) {
                        float dx = fx - s_mx[j];
                        float dy = fy - s_my[j];
                        float adx = dx < 0 ? -dx : dx;
                        float ady = dy < 0 ? -dy : dy;
                        if (adx > s_dx[j] || ady > s_dy[j]) continue;
                    }
                    float dx = fx - s_mx[j];
                    float dy = fy - s_my[j];
                    float q = gca[gid]*dx*dx + 2.0f*gcb[gid]*dx*dy + gcc[gid]*dy*dy;
                    float op = gop[gid];
                    float alpha;
                    if (q > gthr[gid] + 0.01f) { alpha = 0.0f; }
                    else {
                        alpha = op * expf(-0.5f * q);
                        if (alpha < (1.0f/255.0f)) alpha = 0.0f;
                        else if (alpha > ALPHA_CAP) alpha = ALPHA_CAP;
                    }
                    if (alpha > 0.0f) {
                        float w = T * alpha;
                        Cr += w * gcol[gid*3]; Cg += w * gcol[gid*3+1]; Cb += w * gcol[gid*3+2];
                        T *= (1.0f - alpha);
                    }
                }
            }
        }
        __syncthreads();
    }
    if (inside) {
        int p = py * W + px;
        out_img[p*3+0] = Cr; out_img[p*3+1] = Cg; out_img[p*3+2] = Cb;
    }
}

std::vector<torch::Tensor> walk2(
    torch::Tensor gmx, torch::Tensor gmy, torch::Tensor gca, torch::Tensor gcb,
    torch::Tensor gcc, torch::Tensor gop, torch::Tensor gthr, torch::Tensor gcol,
    torch::Tensor gdx, torch::Tensor gdy,
    torch::Tensor gids, torch::Tensor tile_ptr,
    int num_tiles_x, int W, int H, int mode)
{
    int n_pix = W * H;
    auto out = torch::zeros({n_pix, 3}, gmx.options());
    int num_tiles = (int)tile_ptr.numel() - 1;
    dim3 block(TS, TS);
    dim3 grid(num_tiles);
    size_t shmem = (size_t)NARG * CHUNK * sizeof(float);
    walk_kernel2<<<grid, block, shmem>>>(
        gmx.data_ptr<float>(), gmy.data_ptr<float>(),
        gca.data_ptr<float>(), gcb.data_ptr<float>(), gcc.data_ptr<float>(),
        gop.data_ptr<float>(), gthr.data_ptr<float>(), gcol.data_ptr<float>(),
        gdx.data_ptr<float>(), gdy.data_ptr<float>(),
        gids.data_ptr<int>(), tile_ptr.data_ptr<int>(),
        num_tiles_x, W, H, mode, out.data_ptr<float>());
    return {out};
}
