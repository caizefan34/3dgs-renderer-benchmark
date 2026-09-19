#pragma once

// R6-B owns only the five rasterizer backward outputs.  They have the same
// row identity as flatten_ids in unpacked 3DGS ([C, N, ...]).

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <vector>

namespace gsplat::r6b {

enum class Mode : int { Disabled = 0, FullClear = 1, SelectiveClear = 2 };

// One logical row is one unpacked [camera, Gaussian] pair.  Duplicates in
// flatten_ids are harmless: every racing store writes the same zero value.
template <typename T>
__global__ void clear_previous_rows_kernel(
    const int32_t* ids,
    int64_t n_ids,
    T* means2d,
    T* conics,
    T* colors,
    T* opacities,
    T* means2d_abs,
    bool absgrad
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_ids) return;
    const int64_t row = static_cast<int64_t>(ids[i]);
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

class RasterBufferManager {
public:
    void set_mode(int mode) {
        TORCH_CHECK(mode >= 0 && mode <= 2, "R6-B mode must be 0, 1, or 2");
        mode_ = static_cast<Mode>(mode);
        invalidate();
    }

    int mode() const { return static_cast<int>(mode_); }

    float last_prepare_ms() const {
        if (prepare_start_ == nullptr || prepare_end_ == nullptr) return -1.0F;
        float elapsed = 0.0F;
        TORCH_CHECK(cudaEventElapsedTime(&elapsed, prepare_start_, prepare_end_) == cudaSuccess,
                    "R6-B CUDA-event elapsed-time query failed");
        return elapsed;
    }

    void invalidate() {
        force_full_clear_ = true;
        prev_touched_ = torch::Tensor();
    }

    bool enabled_for(const torch::Tensor& means2d) const {
        // Packed tensors do not have the stable [C, N] row identity required
        // by flatten_ids.  The caller must retain the baseline allocation path.
        return mode_ != Mode::Disabled && means2d.dim() == 3;
    }

    std::array<torch::Tensor, 5> prepare(
        const torch::Tensor& means2d,
        const torch::Tensor& conics,
        const torch::Tensor& colors,
        const torch::Tensor& opacities,
        bool absgrad
    ) {
        record_prepare_start();
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

        if (mode_ == Mode::FullClear || force_full_clear_) {
            full_clear();
        } else {
            selective_clear();
        }
        force_full_clear_ = false;
        record_prepare_end();
        return outputs_;
    }

    // Keep immutable forward metadata until the next backward.  This avoids a
    // device-side clone and gives the next iteration exactly its stale rows.
    void finish(const torch::Tensor& flatten_ids) { prev_touched_ = flatten_ids; }

private:
    torch::Tensor active_view(size_t index, const torch::Tensor& reference) {
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

    void full_clear() {
        for (const auto& output : outputs_) {
            if (output.defined()) output.zero_();
        }
    }

    void record_prepare_start() {
        if (prepare_start_ == nullptr) {
            TORCH_CHECK(cudaEventCreate(&prepare_start_) == cudaSuccess,
                        "R6-B failed to create CUDA start event");
            TORCH_CHECK(cudaEventCreate(&prepare_end_) == cudaSuccess,
                        "R6-B failed to create CUDA end event");
        }
        TORCH_CHECK(cudaEventRecord(prepare_start_, at::cuda::getCurrentCUDAStream()) == cudaSuccess,
                    "R6-B failed to record CUDA start event");
    }

    void record_prepare_end() {
        TORCH_CHECK(cudaEventRecord(prepare_end_, at::cuda::getCurrentCUDAStream()) == cudaSuccess,
                    "R6-B failed to record CUDA end event");
    }

    void selective_clear() {
        if (!prev_touched_.defined() || prev_touched_.numel() == 0) return;
        TORCH_CHECK(prev_touched_.scalar_type() == torch::kInt,
                    "R6-B expects int32 flatten_ids");
        constexpr int threads = 256;
        const int blocks = static_cast<int>((prev_touched_.numel() + threads - 1) / threads);
        clear_previous_rows_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            prev_touched_.data_ptr<int32_t>(), prev_touched_.numel(),
            outputs_[0].data_ptr<float>(), outputs_[1].data_ptr<float>(),
            outputs_[2].data_ptr<float>(), outputs_[3].data_ptr<float>(),
            absgrad_ ? outputs_[4].data_ptr<float>() : nullptr, absgrad_
        );
    }

    Mode mode_ = Mode::Disabled;
    bool absgrad_ = false;
    bool force_full_clear_ = true;
    std::array<torch::Tensor, 5> storage_;
    std::array<torch::Tensor, 5> outputs_;
    std::array<std::vector<int64_t>, 5> shapes_;
    torch::Tensor prev_touched_;
    cudaEvent_t prepare_start_ = nullptr;
    cudaEvent_t prepare_end_ = nullptr;
};

inline RasterBufferManager& raster_buffers() {
    static RasterBufferManager manager;
    return manager;
}

void set_mode(int mode) { raster_buffers().set_mode(mode); }
int mode() { return raster_buffers().mode(); }
void invalidate() { raster_buffers().invalidate(); }
float last_prepare_ms() { return raster_buffers().last_prepare_ms(); }

} // namespace gsplat::r6b
