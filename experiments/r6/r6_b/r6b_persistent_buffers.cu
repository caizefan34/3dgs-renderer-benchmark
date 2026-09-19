// R6-B persistent buffer implementation — compiled by nvcc only.
// Contains CUDA kernels and the RasterBufferManager member functions that
// launch them.  Declarations are in r6b_persistent_buffers.cuh.

#include "r6b_persistent_buffers.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <vector>

namespace gsplat::r6b {

// ---------------------------------------------------------------------------
// B1-v2 phase 1: build a [num_rows] touched mask from flatten_ids via scatter.
// ---------------------------------------------------------------------------
__global__ void build_touched_mask_kernel(
    const int32_t* flatten_ids,
    int64_t n_isects,
    bool* mask
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_isects) return;
    mask[flatten_ids[i]] = true;
}

// ---------------------------------------------------------------------------
// B1-v2 phase 2: scan the [num_rows] mask once; clear the 11 rasterizer
// gradient floats for a row ONLY if that row was previously touched.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void clear_touched_rows_kernel(
    const bool* touched_mask,
    int64_t n_rows,
    T* means2d,
    T* conics,
    T* colors,
    T* opacities,
    T* means2d_abs,
    bool absgrad
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    if (!touched_mask[row]) return;
    means2d[2 * row] = 0;
    means2d[2 * row + 1] = 0;
    conics[3 * row] = 0;
    conics[3 * row + 1] = 0;
    conics[3 * row + 2] = 0;
    colors[3 * row] = 0;
    colors[3 * row + 1] = 0;
    colors[3 * row + 2] = 0;
    opacities[row] = 0;
    if (absgrad) {
        means2d_abs[2 * row] = 0;
        means2d_abs[2 * row + 1] = 0;
    }
}

// ---------------------------------------------------------------------------
// RasterBufferManager member functions
// ---------------------------------------------------------------------------

void RasterBufferManager::set_mode(int mode) {
    TORCH_CHECK(mode >= 0 && mode <= 2, "R6-B mode must be 0, 1, or 2");
    mode_ = static_cast<Mode>(mode);
    invalidate();
}

int RasterBufferManager::mode() const { return static_cast<int>(mode_); }

float RasterBufferManager::last_prepare_ms() const {
    return event_ms(prepare_start_, prepare_end_);
}

float RasterBufferManager::last_scatter_ms() const {
    return event_ms(scatter_start_, scatter_end_);
}

float RasterBufferManager::last_clear_ms() const {
    return event_ms(clear_start_, clear_end_);
}

int64_t RasterBufferManager::metadata_bytes() const {
    return prev_touched_.defined() ? prev_touched_.nbytes() : 0;
}

int64_t RasterBufferManager::prev_n_rows() const {
    return prev_touched_.defined() ? prev_touched_.numel() : 0;
}

void RasterBufferManager::invalidate() {
    force_full_clear_ = true;
    prev_touched_ = torch::Tensor();
}

bool RasterBufferManager::enabled_for(const torch::Tensor& means2d) const {
    return mode_ != Mode::Disabled && means2d.dim() == 3;
}

std::array<torch::Tensor, 5> RasterBufferManager::prepare(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    bool absgrad
) {
    record_event(prepare_start_);
    const std::array<torch::Tensor, 5> refs = {
        means2d, conics, colors, opacities, absgrad ? means2d : torch::Tensor()
    };
    bool layout_changed = absgrad != absgrad_;
    for (size_t i = 0; i < refs.size(); ++i) {
        if (!refs[i].defined()) continue;
        const auto shape = refs[i].sizes().vec();
        if (shapes_[i] != shape) layout_changed = true;
        shapes_[i] = shape;
        outputs_[i] = active_view(i, refs[i]);
    }
    absgrad_ = absgrad;
    if (layout_changed) invalidate();

    record_event(clear_start_);
    if (mode_ == Mode::FullClear || force_full_clear_) {
        full_clear();
    } else {
        selective_clear();
    }
    record_event(clear_end_);
    force_full_clear_ = false;
    record_event(prepare_end_);
    return outputs_;
}

void RasterBufferManager::finish(const torch::Tensor& flatten_ids, int64_t num_rows) {
    if (mode_ != Mode::SelectiveClear) return;
    if (num_rows <= 0) return;
    if (!prev_touched_.defined() || prev_touched_.numel() != num_rows ||
        prev_touched_.device() != flatten_ids.device()) {
        prev_touched_ = torch::zeros({num_rows},
                                     flatten_ids.options().dtype(torch::kBool));
    } else {
        prev_touched_.zero_();
    }
    if (flatten_ids.numel() == 0) return;
    TORCH_CHECK(flatten_ids.scalar_type() == torch::kInt,
                "R6-B expects int32 flatten_ids");
    record_event(scatter_start_);
    constexpr int threads = 256;
    const int blocks = static_cast<int>(
        (flatten_ids.numel() + threads - 1) / threads);
    build_touched_mask_kernel<<<blocks, threads, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
        flatten_ids.data_ptr<int32_t>(), flatten_ids.numel(),
        prev_touched_.data_ptr<bool>()
    );
    record_event(scatter_end_);
}

// ---- private helpers ----

torch::Tensor RasterBufferManager::active_view(size_t index, const torch::Tensor& reference) {
    auto& storage = storage_[index];
    const int64_t required = reference.numel();
    const bool incompatible = !storage.defined() ||
        storage.device() != reference.device() ||
        storage.scalar_type() != reference.scalar_type() ||
        storage.numel() < required;
    if (incompatible) {
        const int64_t grown = storage.defined() ? storage.numel() + storage.numel() / 2 : 0;
        storage = torch::empty({std::max(required, grown)}, reference.options());
        force_full_clear_ = true;
    }
    return storage.narrow(0, 0, required).view(reference.sizes());
}

void RasterBufferManager::full_clear() {
    for (const auto& output : outputs_) {
        if (output.defined()) output.zero_();
    }
}

void RasterBufferManager::selective_clear() {
    if (!prev_touched_.defined() || prev_touched_.numel() == 0) return;
    TORCH_CHECK(prev_touched_.scalar_type() == torch::kBool,
                "R6-B-v2 expects a bool touched mask");
    constexpr int threads = 256;
    const int blocks = static_cast<int>(
        (prev_touched_.numel() + threads - 1) / threads);
    clear_touched_rows_kernel<float><<<blocks, threads, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
        prev_touched_.data_ptr<bool>(), prev_touched_.numel(),
        outputs_[0].data_ptr<float>(), outputs_[1].data_ptr<float>(),
        outputs_[2].data_ptr<float>(), outputs_[3].data_ptr<float>(),
        absgrad_ ? outputs_[4].data_ptr<float>() : nullptr, absgrad_
    );
}

void RasterBufferManager::record_event(cudaEvent_t& ev) {
    ensure_event(ev);
    TORCH_CHECK(cudaEventRecord(ev, at::cuda::getCurrentCUDAStream()) == cudaSuccess,
                "R6-B failed to record CUDA event");
}

void RasterBufferManager::ensure_event(cudaEvent_t& ev) {
    if (ev == nullptr) {
        TORCH_CHECK(cudaEventCreate(&ev) == cudaSuccess,
                    "R6-B failed to create CUDA event");
    }
}

float RasterBufferManager::event_ms(cudaEvent_t start, cudaEvent_t end) const {
    if (start == nullptr || end == nullptr) return -1.0F;
    float elapsed = 0.0F;
    TORCH_CHECK(cudaEventElapsedTime(&elapsed, start, end) == cudaSuccess,
                "R6-B CUDA-event elapsed-time query failed");
    return elapsed;
}

// ---- free functions ----

void set_mode(int mode) { raster_buffers().set_mode(mode); }
int mode() { return raster_buffers().mode(); }
void invalidate() { raster_buffers().invalidate(); }
float last_prepare_ms() { return raster_buffers().last_prepare_ms(); }
float last_scatter_ms() { return raster_buffers().last_scatter_ms(); }
float last_clear_ms() { return raster_buffers().last_clear_ms(); }
int64_t metadata_bytes() { return raster_buffers().metadata_bytes(); }
int64_t prev_n_rows() { return raster_buffers().prev_n_rows(); }

} // namespace gsplat::r6b
