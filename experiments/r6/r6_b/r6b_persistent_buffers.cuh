#pragma once

// R6-B owns only the five rasterizer backward outputs.  They have the same
// row identity as flatten_ids in unpacked 3DGS ([C, N, ...]).
//
// B0  (FullClear):       persistent capacity-managed buffers + full active-range zero-fill.
// B1-v1 (SUPERSEDED):    stored flatten_ids [n_isects] (intersection list with duplicates)
//                        and launched one clear per intersection — O(n_isects) gradient writes.
// B1-v2 (SelectiveClear): scatters flatten_ids into a compact [C*N] bool touched mask
//                        (O(n_isects) 1-byte writes), then scans the mask once and clears
//                        only touched gradient rows — O(C*N predicate reads + N_touched
//                        gradient writes).  No unique / sort / cumsum / CPU sync.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <vector>

namespace gsplat::r6b {

enum class Mode : int { Disabled = 0, FullClear = 1, SelectiveClear = 2 };

// ---------------------------------------------------------------------------
// B1-v2 phase 1: build a [num_rows] touched mask from flatten_ids via scatter.
// Each thread sets mask[flatten_ids[i]] = true.  Duplicate IDs are idempotent
// (every racing store writes the same value), so no atomics are required.
// Complexity: O(n_isects) one-byte writes — NOT unique / sort / cumsum.
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
// Complexity: O(num_rows predicate reads + N_touched * 11 gradient writes).
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
    if (!touched_mask[row]) return;  // skip untouched rows — the whole point
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

    // ---- timing accessors -------------------------------------------------
    float last_prepare_ms() const { return event_ms(prepare_start_, prepare_end_); }
    // B1-v2: mask-build (scatter) time, recorded inside finish().
    float last_scatter_ms() const { return event_ms(scatter_start_, scatter_end_); }
    // B1-v2: selective-clear (mask-scan + gradient-zero) time, recorded inside prepare().
    // For B0 this equals last_prepare_ms (the only work in prepare is the full clear).
    float last_clear_ms() const { return event_ms(clear_start_, clear_end_); }

    // Extra persistent metadata: the [C*N] bool touched mask.
    int64_t metadata_bytes() const {
        return prev_touched_.defined() ? prev_touched_.nbytes() : 0;
    }

    int64_t prev_n_rows() const {
        return prev_touched_.defined() ? prev_touched_.numel() : 0;
    }

    void invalidate() {
        force_full_clear_ = true;
        prev_touched_ = torch::Tensor();
    }

    bool enabled_for(const torch::Tensor& means2d) const {
        // Packed tensors do not have the stable [C, N] row identity required
        // by the touched mask.  The caller must retain the baseline allocation path.
        return mode_ != Mode::Disabled && means2d.dim() == 3;
    }

    std::array<torch::Tensor, 5> prepare(
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

    // B1-v2: Build a [num_rows] touched mask from flatten_ids via a scatter
    // kernel.  The mask is O(num_rows) bytes — far smaller than the
    // O(n_isects * 4) bytes of flatten_ids retained in B1-v1 — and is
    // semantically equivalent to (tiles_per_gauss > 0).reshape(-1).
    void finish(const torch::Tensor& flatten_ids, int64_t num_rows) {
        if (mode_ != Mode::SelectiveClear) return;
        if (num_rows <= 0) return;
        // Reuse or allocate the persistent mask buffer (avoids per-iteration alloc).
        if (!prev_touched_.defined() || prev_touched_.numel() != num_rows ||
            prev_touched_.device() != flatten_ids.device()) {
            prev_touched_ = torch::zeros({num_rows},
                                         flatten_ids.options().dtype(torch::kBool));
        } else {
            prev_touched_.zero_();  // reset before scatter
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

    void selective_clear() {
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

    // ---- CUDA event helpers -----------------------------------------------
    void ensure_events(cudaEvent_t& a, cudaEvent_t& b) {
        if (a == nullptr) {
            TORCH_CHECK(cudaEventCreate(&a) == cudaSuccess,
                        "R6-B failed to create CUDA event");
            TORCH_CHECK(cudaEventCreate(&b) == cudaSuccess,
                        "R6-B failed to create CUDA event");
        }
    }

    void record_event(cudaEvent_t& ev) {
        ensure_event_pair(ev);
        TORCH_CHECK(cudaEventRecord(ev, at::cuda::getCurrentCUDAStream()) == cudaSuccess,
                    "R6-B failed to record CUDA event");
    }

    float event_ms(cudaEvent_t start, cudaEvent_t end) const {
        if (start == nullptr || end == nullptr) return -1.0F;
        float elapsed = 0.0F;
        TORCH_CHECK(cudaEventElapsedTime(&elapsed, start, end) == cudaSuccess,
                    "R6-B CUDA-event elapsed-time query failed");
        return elapsed;
    }

    void ensure_event_pair(cudaEvent_t& ev) {
        // Each event is self-contained; create lazily on first use.
        if (ev == nullptr) {
            TORCH_CHECK(cudaEventCreate(&ev) == cudaSuccess,
                        "R6-B failed to create CUDA event");
        }
    }

    // ---- state ------------------------------------------------------------
    Mode mode_ = Mode::Disabled;
    bool absgrad_ = false;
    bool force_full_clear_ = true;
    std::array<torch::Tensor, 5> storage_;
    std::array<torch::Tensor, 5> outputs_;
    std::array<std::vector<int64_t>, 5> shapes_;
    torch::Tensor prev_touched_;  // B1-v2: [C*N] bool touched mask

    cudaEvent_t prepare_start_ = nullptr;
    cudaEvent_t prepare_end_ = nullptr;
    cudaEvent_t scatter_start_ = nullptr;
    cudaEvent_t scatter_end_ = nullptr;
    cudaEvent_t clear_start_ = nullptr;
    cudaEvent_t clear_end_ = nullptr;
};

inline RasterBufferManager& raster_buffers() {
    static RasterBufferManager manager;
    return manager;
}

void set_mode(int mode) { raster_buffers().set_mode(mode); }
int mode() { return raster_buffers().mode(); }
void invalidate() { raster_buffers().invalidate(); }
float last_prepare_ms() { return raster_buffers().last_prepare_ms(); }
float last_scatter_ms() { return raster_buffers().last_scatter_ms(); }
float last_clear_ms() { return raster_buffers().last_clear_ms(); }
int64_t metadata_bytes() { return raster_buffers().metadata_bytes(); }
int64_t prev_n_rows() { return raster_buffers().prev_n_rows(); }

} // namespace gsplat::r6b
