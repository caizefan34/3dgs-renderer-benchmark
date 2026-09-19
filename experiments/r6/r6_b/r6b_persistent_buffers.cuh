#pragma once

// R6-B: declarations only.  Implementations (including CUDA kernels) are in
// r6b_persistent_buffers.cu, compiled by nvcc.  This header is safe to include
// from .cpp files compiled by the host compiler.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <array>
#include <vector>

namespace gsplat::r6b {

enum class Mode : int { Disabled = 0, FullClear = 1, SelectiveClear = 2 };

class RasterBufferManager {
public:
    void set_mode(int mode);
    int mode() const;

    float last_prepare_ms() const;
    float last_scatter_ms() const;
    float last_clear_ms() const;

    int64_t metadata_bytes() const;
    int64_t prev_n_rows() const;

    void invalidate();

    bool enabled_for(const torch::Tensor& means2d) const;

    std::array<torch::Tensor, 5> prepare(
        const torch::Tensor& means2d,
        const torch::Tensor& conics,
        const torch::Tensor& colors,
        const torch::Tensor& opacities,
        bool absgrad
    );

    void finish(const torch::Tensor& flatten_ids, int64_t num_rows);

private:
    // All state is in the .cu file; this class is just an interface.
    // We use pImpl-like storage via void* to avoid CUDA types in the header.
    // Actually, since torch::Tensor is fine in host code, we keep the state
    // members here.  The kernels are only in the .cu file.
    Mode mode_ = Mode::Disabled;
    bool absgrad_ = false;
    bool force_full_clear_ = true;
    std::array<torch::Tensor, 5> storage_;
    std::array<torch::Tensor, 5> outputs_;
    std::array<std::vector<int64_t>, 5> shapes_;
    torch::Tensor prev_touched_;

    // CUDA events — these are just opaque pointers in host code.
    // We store them as cudaEvent_t which requires the CUDA runtime header
    // (already included above via ATen/cuda/CUDAContext.h).
    cudaEvent_t prepare_start_ = nullptr;
    cudaEvent_t prepare_end_ = nullptr;
    cudaEvent_t scatter_start_ = nullptr;
    cudaEvent_t scatter_end_ = nullptr;
    cudaEvent_t clear_start_ = nullptr;
    cudaEvent_t clear_end_ = nullptr;

    // Private helpers declared here, implemented in .cu
    torch::Tensor active_view(size_t index, const torch::Tensor& reference);
    void full_clear();
    void selective_clear();
    void record_event(cudaEvent_t& ev);
    void ensure_event(cudaEvent_t& ev);
    float event_ms(cudaEvent_t start, cudaEvent_t end) const;
};

inline RasterBufferManager& raster_buffers() {
    static RasterBufferManager manager;
    return manager;
}

void set_mode(int mode);
int mode();
void invalidate();
float last_prepare_ms();
float last_scatter_ms();
float last_clear_ms();
int64_t metadata_bytes();
int64_t prev_n_rows();

} // namespace gsplat::r6b
