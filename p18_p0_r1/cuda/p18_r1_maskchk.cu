// P18-P0-R1 trace-record verifier: recompute each dumped record's ok_mask for ALL
// 16 tile rows with the EXACT half2 code path of MacroTileRasterize.cu phase 2 and
// emit the full hamming-distance matrix d16[n,16] (distance of row rr's mask from
// the stored mask16). Row analysis (claimed row, pp-candidates, best row) is done
// host-side in numpy so no GPU re-runs are needed.
//
// Decoded record (8 x u32): [0] g  [1] mt_batch  [2] x0|y0<<16  [3] qidx|mask16<<16
//                          [4] pos.x bits [5] pos.y bits [6] conic01 [7] conic23
// conic halves: {l0, l1, l2, opacity}; geometry: mt_col = x0/128, mt_row = y0/64,
// tl_col = (x0/16)%8, tl_row = (y0/16)%4, row = y0%16 (= ty + 4pp + 2h in the slot).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace {
constexpr float INV_TILE_SIZE = 1.0f / 16.0f;
constexpr float CHOL_SCALE = 0.84932180028f * 16.0f; // sqrt(LOG2E/2) * TILE_SIZE
constexpr float LOG2_INV_AT = 7.9943534f;
constexpr float FUSED_MW = 8.0f, FUSED_MH = 4.0f;

struct Chk
{
    const uint32_t *__restrict__ rec; // [n*8]
    int n;
    unsigned char *__restrict__ d16;  // [n*16]
};

__device__ __forceinline__ uint32_t mask_of_row(
    float pos_x, float pos_y, const __half2 (&raw)[2],
    int mt_col, int mt_row, int tl_col, int tl_row, int row_in_tile)
{
    const int ty   = row_in_tile & 1;
    const int q    = row_in_tile >> 1; // 2*pp + h
    const __half2 xy = __floats2half2_rn(pos_x * INV_TILE_SIZE - (mt_col * FUSED_MW + FUSED_MW * 0.5f),
                                         pos_y * INV_TILE_SIZE - (mt_row * FUSED_MH + FUSED_MH * 0.5f));
    // exact replication of kernel phase-1 conic transform (MacroTileRasterize.cu L499-506):
    // rc[0] = __hmul2(raw[0], half(CHOL_SCALE)); rc[1] = {f2h(l2_fp32 * CHOL_SCALE), f2h(-__log2f(opac))}
    __half2 rc[2];
    rc[0] = __hmul2(raw[0], __floats2half2_rn(CHOL_SCALE, CHOL_SCALE));
    const float opacity = __half2float(__high2half(raw[1]));
    rc[1] = __floats2half2_rn(__half2float(__low2half(raw[1])) * CHOL_SCALE, -__log2f(opacity));
    const __half2 l00          = __low2half2(rc[0]);
    const __half2 l10          = __high2half2(rc[0]);
    const __half2 l11          = __low2half2(rc[1]);
    const __half2 neg_log_opac = __high2half2(rc[1]);
    const __half2 sigma_thresh = __float2half2_rn(LOG2_INV_AT);
    const __half2 mean_y       = __high2half2(xy);
    const __half2 base_y = __float2half2_rn(static_cast<float>(tl_row) + ty * INV_TILE_SIZE
                                            + (0.5f * INV_TILE_SIZE - 0.5f) + (0.5f - FUSED_MH * 0.5f));
    const __half2 offset_y = __hadd2(base_y, __float2half2_rn(q * 2 * INV_TILE_SIZE));
    const __half2 dy = __hsub2(mean_y, offset_y);

    uint32_t m = 0;
#pragma unroll
    for (int tx = 0; tx < 16; ++tx)
    {
        const float px = static_cast<float>(tx) * INV_TILE_SIZE
                         + (0.5f * INV_TILE_SIZE - 0.5f) + (0.5f - FUSED_MW * 0.5f);
        const __half2 offset_x = __float2half2_rn(static_cast<float>(tl_col) + px);
        const __half2 dx       = __hsub2(__low2half2(xy), offset_x);
        const __half2 u0       = __hfma2(l10, dy, __hmul2(l00, dx));
        const __half2 u1       = __hmul2(l11, dy);
        const __half2 sigma    = __hfma2(u1, u1, __hfma2(u0, u0, neg_log_opac));
        if(__hlt2_mask(sigma, sigma_thresh) & 1u) m |= (1u << tx);
    }
    return m;
}

__global__ void maskchk_kernel(Chk c)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= c.n) return;
    const uint32_t *r = c.rec + static_cast<size_t>(i) * 8;
    const uint32_t xy    = r[2];
    const uint32_t stored = r[3] >> 16;
    const int x0 = static_cast<int>(xy & 0xFFFFu);
    const int y0 = static_cast<int>(xy >> 16);
    const float pos_x = __uint_as_float(r[4]);
    const float pos_y = __uint_as_float(r[5]);
    __half2 raw[2];
    raw[0] = *reinterpret_cast<const __half2 *>(&r[6]);
    raw[1] = *reinterpret_cast<const __half2 *>(&r[7]);
    const int mt_col = x0 / (8 * 16);
    const int mt_row = y0 / (4 * 16);
    const int tl_col = (x0 / 16) % 8;
    const int tl_row = (y0 / 16) % 4;
    unsigned char *dst = c.d16 + static_cast<size_t>(i) * 16;
#pragma unroll
    for (int rr = 0; rr < 16; ++rr)
    {
        const uint32_t m = mask_of_row(pos_x, pos_y, raw, mt_col, mt_row, tl_col, tl_row, rr);
        dst[rr] = static_cast<unsigned char>(__popc(m ^ stored));
    }
}
} // namespace

std::vector<torch::Tensor> maskchk(torch::Tensor rec, int64_t n)
{
    auto opts_b = torch::TensorOptions().dtype(torch::kUInt8).device(rec.device());
    auto d16 = torch::zeros({n, 16}, opts_b);
    Chk c{reinterpret_cast<const uint32_t *>(rec.data_ptr<int32_t>()), static_cast<int>(n),
          d16.data_ptr<unsigned char>()};
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 128, blocks = (n + threads - 1) / threads;
    maskchk_kernel<<<blocks, threads, 0, stream>>>(c);
    return {d16};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("maskchk", &maskchk);
}
