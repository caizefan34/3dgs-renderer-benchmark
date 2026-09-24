// Contract adapter for P2-1A-R2.  This file owns only ABI conversion from the
// shared F9 FP32 projected state into the native HiGS float4 records.

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "IntersectMTFused.h"
#include "ProjectedStateAdapter.h"

namespace gsplat
{
namespace gaussian_render_inference_scene
{
namespace
{
__global__ void radii_to_visible_words_kernel(
    const int32_t *__restrict__ radii, int64_t n, uint32_t *__restrict__ visible_words
)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n && radii[i * 2] > 0 && radii[i * 2 + 1] > 0)
    {
        atomicOr(visible_words + (i >> 5), 1u << (i & 31));
    }
}

void check_projected_tensor(
    const at::Tensor &tensor, at::ScalarType dtype, at::IntArrayRef shape, const char *name
)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.sizes() == shape, name, " has an invalid shape");
}
} // namespace

std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> higs_native_hierarchy_from_projected(
    const at::Tensor &visible_ids,
    const at::Tensor &radii,
    const at::Tensor &means2d,
    const at::Tensor &depths,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &colors,
    int64_t width,
    int64_t height,
    int64_t tile_size,
    const at::optional<at::Tensor> &background,
    bool debug
)
{
    TORCH_CHECK(width > 0 && height > 0, "width and height must be positive");
    TORCH_CHECK(tile_size == 8 || tile_size == 16, "native hierarchy supports tile_size 8 or 16");
    TORCH_CHECK(visible_ids.is_cuda() && visible_ids.scalar_type() == at::kLong && visible_ids.dim() == 1,
                "visible_ids must be contiguous CUDA int64 [N]");
    TORCH_CHECK(visible_ids.is_contiguous(), "visible_ids must be contiguous");

    const int64_t n = visible_ids.size(0);
    check_projected_tensor(radii, at::kInt, {n, 2}, "radii");
    check_projected_tensor(means2d, at::kFloat, {n, 2}, "means2d");
    check_projected_tensor(depths, at::kFloat, {n}, "depths");
    check_projected_tensor(conics, at::kFloat, {n, 3}, "conics");
    check_projected_tensor(opacities, at::kFloat, {n}, "opacities");
    check_projected_tensor(colors, at::kFloat, {n, 3}, "colors");
    TORCH_CHECK(
        visible_ids.device() == means2d.device() && radii.device() == means2d.device()
            && depths.device() == means2d.device() && conics.device() == means2d.device()
            && opacities.device() == means2d.device() && colors.device() == means2d.device(),
        "all projected-state tensors must be on the same CUDA device"
    );

    const at::cuda::OptionalCUDAGuard device_guard(means2d.device());
    const auto opts_f = means2d.options().dtype(at::kFloat);
    const auto opts_i = means2d.options().dtype(at::kInt);
    at::Tensor packed_background = at::zeros({1, 4}, opts_f);
    packed_background.select(1, 3).fill_(1.0f);
    if(background.has_value())
    {
        const auto &bg = background.value();
        TORCH_CHECK(bg.is_cuda() && bg.scalar_type() == at::kFloat && bg.is_contiguous() && bg.numel() == 3,
                    "background must be contiguous CUDA FP32 with three elements");
        TORCH_CHECK(bg.device() == means2d.device(), "background must be on the projected-state device");
        packed_background.narrow(1, 0, 3).copy_(bg.reshape({1, 3}));
    }

    at::Tensor rgbt = at::empty({height, width, 4}, opts_f);
    if(n == 0)
    {
        rgbt.zero_();
        rgbt.select(2, 3).fill_(1.0f);
        rgbt.narrow(2, 0, 3).copy_(packed_background.narrow(1, 0, 3).reshape({1, 1, 3}));
        return {rgbt.narrow(2, 0, 3).contiguous(), at::zeros({height, width, 1}, opts_f), {}};
    }

    // The only state transformation: F9 supplies the FP32 symmetric inverse
    // covariance {ci00, ci01, ci11}; the native hierarchy's ABI is its lower
    // Cholesky factor {l0, l1, l2}, where Sigma^-1 = L*L^T.  This exactly
    // mirrors Projection.cu's FP32 factorization but performs no projection,
    // SH, opacity, or FP16 conversion.
    const at::Tensor l0 = at::sqrt(at::clamp_min(conics.select(1, 0), 0.0));
    const at::Tensor l1 = at::where(
        at::gt(l0, 1.0e-12), at::div(conics.select(1, 1), l0), at::zeros_like(l0)
    );
    const at::Tensor l2 = at::sqrt(at::clamp_min(at::sub(conics.select(1, 2), at::mul(l1, l1)), 0.0));
    at::Tensor packed_conics = at::zeros({n, 4}, opts_f);
    packed_conics.select(1, 0).copy_(l0);
    packed_conics.select(1, 1).copy_(l1);
    packed_conics.select(1, 2).copy_(l2);
    packed_conics.select(1, 3).copy_(opacities);
    at::Tensor packed_colors = at::zeros({n, 4}, opts_f);
    packed_colors.narrow(1, 0, 3).copy_(colors);

    at::Tensor visible_words = at::zeros({(n + 31) / 32}, opts_i);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    radii_to_visible_words_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        radii.data_ptr<int32_t>(), n, reinterpret_cast<uint32_t *>(visible_words.data_ptr<int32_t>())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int32_t tile_width = static_cast<int32_t>((width + tile_size - 1) / tile_size);
    const int32_t tile_height = static_cast<int32_t>((height + tile_size - 1) / tile_size);
    IntersectMTFused hierarchy;
    hierarchy.execute(
        means2d, depths, packed_conics, visible_words, static_cast<int32_t>(tile_size), tile_width, tile_height,
        at::cuda::getCurrentCUDAStream()
    );
    hierarchy.rasterize(means2d, packed_conics, packed_colors, packed_background,
                         static_cast<uint32_t>(width), static_cast<uint32_t>(height), rgbt);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<at::Tensor> diagnostics;
    if(debug)
    {
        // These native buffers, plus the returned F9 state, recover macro entries,
        // macro batches, 32-G mini-batches and exact per-fine-tile traversal without
        // adding instrumentation writes to the authoritative raster launch.
        diagnostics = hierarchy.debug_intersection_tensors();
        diagnostics.push_back(visible_ids);
        diagnostics.push_back(depths);
    }
    return {rgbt.narrow(2, 0, 3).contiguous(),
            at::ones_like(rgbt.narrow(2, 3, 1)).sub_(rgbt.narrow(2, 3, 1)).contiguous(), diagnostics};
}
} // namespace gaussian_render_inference_scene
} // namespace gsplat
