#pragma once

// R6-B owns only the five rasterizer backward outputs.  They have the same
// row identity as flatten_ids in unpacked 3DGS ([C, N, ...]).

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <vector>

namespace gsplat::r6b {

enum class Mode : int { Disabled = 0, FullClear = 1 };

class RasterBufferManager {
public:
    void set_mode(int mode) {
        TORCH_CHECK(mode >= 0 && mode <= 1, "R6-B mode must be 0 or 1");
        mode_ = static_cast<Mode>(mode);
        invalidate();
    }

    int mode() const { return static_cast<int>(mode_); }

    void invalidate() {
        force_full_clear_ = true;
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

        full_clear();
        force_full_clear_ = false;
        return outputs_;
    }

    void finish(const torch::Tensor&) {}

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

    Mode mode_ = Mode::Disabled;
    bool absgrad_ = false;
    bool force_full_clear_ = true;
    std::array<torch::Tensor, 5> storage_;
    std::array<torch::Tensor, 5> outputs_;
    std::array<std::vector<int64_t>, 5> shapes_;
};

inline RasterBufferManager& raster_buffers() {
    static RasterBufferManager manager;
    return manager;
}

inline void set_mode(int mode) { raster_buffers().set_mode(mode); }
inline int mode() { return raster_buffers().mode(); }
inline void invalidate() { raster_buffers().invalidate(); }

} // namespace gsplat::r6b
