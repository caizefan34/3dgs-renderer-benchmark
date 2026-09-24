#!/usr/bin/env python3
"""Create an isolated B1A_R6A gsplat tree.

The source B1A tree is copied, never edited.  The candidate replaces only the
rasterizer's warp-leader global-gradient writes with an exact, uniform
block-level reduction.  ``--count-atomics`` makes a separate diagnostic build;
its counter is deliberately disabled in timing builds.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


KERNEL = "RasterizeToPixels3DGSBwd.cu"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected exactly one source context, found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-atomics", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    bwd = source / "gsplat" / "cuda" / "csrc" / KERNEL
    ext = source / "gsplat" / "cuda" / "ext.cpp"
    if not bwd.is_file() or not ext.is_file():
        raise SystemExit("expected gsplat v1.5.x source layout")
    if output.exists():
        raise SystemExit(f"refusing to overwrite candidate tree: {output}")

    shutil.copytree(source, output, symlinks=True, ignore=shutil.ignore_patterns("build", "*.so", "__pycache__"))
    bwd = output / "gsplat" / "cuda" / "csrc" / KERNEL
    ext = output / "gsplat" / "cuda" / "ext.cpp"
    text = bwd.read_text(encoding="utf-8")

    marker = '#include "Utils.cuh"\n'
    injected_header = '''#include "Utils.cuh"
#include <cstdint>

// R6-A is a separately built experimental variant.  The timing build leaves
// this disabled; the diagnostic build enables it to count final block writers.
#define R6A_COUNT_ATOMICS %d
'''.replace("%d", "1" if args.count_atomics else "0")
    text = replace_once(text, marker, injected_header, "header")

    namespace_marker = "namespace gsplat {\n\nnamespace cg = cooperative_groups;"
    counter = '''namespace gsplat {

__device__ unsigned long long r6a_global_writer_events = 0ULL;
__device__ unsigned long long r6a_block_reduction_events = 0ULL;

namespace r6a {
void reset_global_writer_events() {
    const unsigned long long zero = 0ULL;
    cudaMemcpyToSymbol(r6a_global_writer_events, &zero, sizeof(zero), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(r6a_block_reduction_events, &zero, sizeof(zero), 0,
                       cudaMemcpyHostToDevice);
}
uint64_t get_global_writer_events() {
    unsigned long long value = 0ULL;
    cudaMemcpyFromSymbol(&value, r6a_global_writer_events, sizeof(value), 0,
                         cudaMemcpyDeviceToHost);
    return static_cast<uint64_t>(value);
}
uint64_t get_block_reduction_events() {
    unsigned long long value = 0ULL;
    cudaMemcpyFromSymbol(&value, r6a_block_reduction_events, sizeof(value), 0,
                         cudaMemcpyDeviceToHost);
    return static_cast<uint64_t>(value);
}
} // namespace r6a

namespace cg = cooperative_groups;'''
    text = replace_once(text, namespace_marker, counter, "counter namespace")

    shared_old = '''    float *rgbs_batch =
        (float *)&conic_batch[block_size]; // [block_size * CDIM]
'''
    shared_new = '''    float *rgbs_batch =
        (float *)&conic_batch[block_size]; // [block_size * CDIM]
    // One row per warp: CDIM RGB + 3 conic + 2 xy + 2 abs-xy + opacity
    // + a nonzero-contributor flag.  The frozen 16x16 path has eight rows.
    float *r6a_partials = &rgbs_batch[block_size * CDIM];
    const uint32_t r6a_num_warps = block_size / 32;
    const uint32_t r6a_partial_stride = CDIM + 9;
'''
    text = replace_once(text, shared_old, shared_new, "shared layout")

    loop_start = '''    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());
'''
    loop_start_new = '''    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const uint32_t r6a_warp_id = warp.meta_group_rank();
'''
    text = replace_once(text, loop_start, loop_start_new, "warp skip state")

    old_loop = '''        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            vec2 delta;
            vec3 conic;
            float vis;

            if (valid) {
                conic = conic_batch[t];
                vec3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                      conic.z * delta.y * delta.y) +
                              conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= 0.999f) {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    };
                    v_xy_local = {
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    };
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [I * N] or [nnz]
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
'''
    new_loop = old_loop.replace(
        "for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;\n             ++t)",
        "for (uint32_t t = 0; t < batch_size; ++t)",
    ).replace(
        '''            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
''',
        '''            // The loop is deliberately block-uniform.  An inactive warp writes
            // zeros, rather than skipping, so every thread reaches both barriers.
            const bool r6a_warp_active = warp.any(valid);
''',
    )
    old_writer = '''            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [I * N] or [nnz]
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
'''
    new_writer = '''            if (warp.thread_rank() == 0) {
                float *partial = r6a_partials + r6a_warp_id * r6a_partial_stride;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    partial[k] = v_rgb_local[k];
                }
                partial[CDIM + 0] = v_conic_local.x;
                partial[CDIM + 1] = v_conic_local.y;
                partial[CDIM + 2] = v_conic_local.z;
                partial[CDIM + 3] = v_xy_local.x;
                partial[CDIM + 4] = v_xy_local.y;
                partial[CDIM + 5] = v_xy_abs_local.x;
                partial[CDIM + 6] = v_xy_abs_local.y;
                partial[CDIM + 7] = v_opacity_local;
                partial[CDIM + 8] = r6a_warp_active ? 1.f : 0.f;
            }

            // The following barriers are legal because every warp executes the
            // uniform t loop.  The second protects the next overwrite of partials.
            block.sync();
            if (r6a_warp_id == 0) {
                const uint32_t lane = warp.thread_rank();
#if R6A_COUNT_ATOMICS
                if (lane == 0) {
                    ::atomicAdd(&r6a_block_reduction_events, 1ULL);
                }
#endif
                float *partial = lane < r6a_num_warps
                    ? r6a_partials + lane * r6a_partial_stride
                    : nullptr;
                float v_rgb_block[CDIM] = {0.f};
                vec3 v_conic_block = {0.f, 0.f, 0.f};
                vec2 v_xy_block = {0.f, 0.f};
                vec2 v_xy_abs_block = {0.f, 0.f};
                float v_opacity_block = 0.f;
                bool r6a_block_active = false;
                if (lane < r6a_num_warps) {
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_rgb_block[k] = partial[k];
                    }
                    v_conic_block = {partial[CDIM], partial[CDIM + 1], partial[CDIM + 2]};
                    v_xy_block = {partial[CDIM + 3], partial[CDIM + 4]};
                    v_xy_abs_block = {partial[CDIM + 5], partial[CDIM + 6]};
                    v_opacity_block = partial[CDIM + 7];
                    r6a_block_active = partial[CDIM + 8] != 0.f;
                }
                warpSum<CDIM>(v_rgb_block, warp);
                warpSum(v_conic_block, warp);
                warpSum(v_xy_block, warp);
                warpSum(v_xy_abs_block, warp);
                warpSum(v_opacity_block, warp);
                r6a_block_active = warp.any(r6a_block_active);
                if (lane == 0 && r6a_block_active) {
                    int32_t g = id_batch[t];
                    float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        gpuAtomicAdd(v_rgb_ptr + k, v_rgb_block[k]);
                    }
                    float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                    gpuAtomicAdd(v_conic_ptr, v_conic_block.x);
                    gpuAtomicAdd(v_conic_ptr + 1, v_conic_block.y);
                    gpuAtomicAdd(v_conic_ptr + 2, v_conic_block.z);
                    float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                    gpuAtomicAdd(v_xy_ptr, v_xy_block.x);
                    gpuAtomicAdd(v_xy_ptr + 1, v_xy_block.y);
                    if (v_means2d_abs != nullptr) {
                        float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                        gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_block.x);
                        gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_block.y);
                    }
                    gpuAtomicAdd(v_opacities + g, v_opacity_block);
#if R6A_COUNT_ATOMICS
                    ::atomicAdd(&r6a_global_writer_events, 1ULL);
#endif
                }
            }
            block.sync();
'''
    new_loop = new_loop.replace(old_writer, new_writer, 1)
    if new_loop == old_loop:
        raise RuntimeError("writer replacement failed")
    text = replace_once(text, old_loop, new_loop, "uniform block loop")

    shmem_old = '''    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);
'''
    shmem_new = '''    int64_t shmem_size =
        tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM) +
        (tile_size * tile_size / 32) * (CDIM + 9) * sizeof(float);
'''
    text = replace_once(text, shmem_old, shmem_new, "shared-memory launch size")
    bwd.write_text(text, encoding="utf-8")

    ext_text = ext.read_text(encoding="utf-8")
    ext_text = replace_once(
        ext_text, '#include <torch/extension.h>\n',
        '#include <torch/extension.h>\n#include <cstdint>\n\nnamespace gsplat::r6a {\nvoid reset_global_writer_events();\nuint64_t get_global_writer_events();\nuint64_t get_block_reduction_events();\n}\n',
        "ext declarations",
    )
    anchor = '''    m.def(
        "rasterize_to_pixels_3dgs_bwd", &gsplat::rasterize_to_pixels_3dgs_bwd
    );'''
    ext_text = replace_once(
        ext_text, anchor,
        anchor + '\n    m.def("r6a_reset_global_writer_events", &gsplat::r6a::reset_global_writer_events);\n'
        '    m.def("r6a_get_global_writer_events", &gsplat::r6a::get_global_writer_events);\n'
        '    m.def("r6a_get_block_reduction_events", &gsplat::r6a::get_block_reduction_events);',
        "ext bindings",
    )
    ext.write_text(ext_text, encoding="utf-8")
    print(f"created {'counter' if args.count_atomics else 'timing'} candidate: {output}")


if __name__ == "__main__":
    main()
