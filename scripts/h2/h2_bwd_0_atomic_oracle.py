#!/usr/bin/env python3
"""H2-BWD-0 Section 7: Atomic-free upper-bound experiment.

Measures the upper bound of eliminating global atomicAdd in the blend backward
by comparing two microbenchmark kernels with identical compute but different
scatter patterns:
  A) atomicAdd to contended global gradient buffers (same as real kernel)
  B) uncontended scratch write (same compute, no atomic)

Both variants reproduce the exact workload structure of higs_blend_bwd_px_kernel<3,2>:
  - Same grid: (1, tile_h, tile_w) = 11008 blocks
  - Same block: (16, 8, 1) = 128 threads = 4 warps
  - Same batch processing of intersections from tile_offsets/flatten_ids
  - Same warpSum reduction
  - Same early-exit via bin_final
  - Same Gaussian weight evaluation (exp, mul, add)
  - Same VJP-like compute (9 gradient components per intersection)

Uses torch.utils.cpp_extension.load_inline for on-the-fly compilation.
"""
import argparse, json, math, os, time, sys
import numpy as np
import torch

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>

namespace cg = cooperative_groups;

// Reproduce higs_blend_bwd_px_kernel<3,2> structure:
// Block: (16, 8, 1) = 128 threads, PX=2, PX_ROWS=8
// Grid: (I, tile_h, tile_w) for dense path
// 4 warps: warp w covers ty in [2w, 2w+1]
// Warp 0: pixel rows {0,1,8,9}, Warp 1: {2,3,10,11}, etc.

template<int CDIM, int PX, bool USE_ATOMIC>
__global__ void scatter_oracle_kernel(
    const uint32_t n_isects,
    const float *__restrict__ means2d_flat,    // [N*2]
    const float *__restrict__ conics_flat,     // [N*3]
    const float *__restrict__ colors_flat,     // [N*CDIM]
    const float *__restrict__ opacities,       // [N]
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const int32_t *__restrict__ tile_offsets,  // [tile_h*tile_w]
    const int32_t *__restrict__ flatten_ids,   // [n_isects]
    const float *__restrict__ render_alphas,   // [H*W]
    const int32_t *__restrict__ last_ids,      // [H*W]
    const float *__restrict__ v_render_colors, // [H*W*CDIM]
    const float *__restrict__ v_render_alphas, // [H*W]
    float *__restrict__ v_colors_out,          // [N*CDIM]
    float *__restrict__ v_conics_out,          // [N*3]
    float *__restrict__ v_means2d_out,         // [N*2]
    float *__restrict__ v_opacities_out,       // [N]
    float *__restrict__ scratch                // [n_isects*4*9] for non-atomic
)
{
    constexpr uint32_t PX_ROWS = 16 / PX;  // 8

    auto block = cg::this_thread_block();
    const uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i0 = block.group_index().y * tile_size;
    const uint32_t j0 = block.group_index().z * tile_size;
    const uint32_t ty = block.thread_index().y;
    const uint32_t tx = block.thread_index().x;

    // Per-pixel state
    float px[PX], py[PX];
    int32_t pix_id[PX];
    bool inside[PX];
    float T_final[PX], T[PX];
    float buffer[PX][CDIM];
    float v_render_c[PX][CDIM];
    float v_render_a[PX];
    int32_t bin_final[PX];

    #pragma unroll
    for(uint32_t q = 0; q < PX; ++q) {
        const uint32_t i = i0 + ty + q * PX_ROWS;
        const uint32_t j = j0 + tx;
        px[q] = (float)j + 0.5f;
        py[q] = (float)i + 0.5f;
        pix_id[q] = min(i * image_width + j, image_width * image_height - 1);
        inside[q] = (i < image_height && j < image_width);
        T_final[q] = 1.0f - render_alphas[pix_id[q]];
        T[q] = T_final[q];
        bin_final[q] = inside[q] ? last_ids[pix_id[q]] : 0;
        #pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k) {
            v_render_c[q][k] = v_render_colors[pix_id[q] * CDIM + k];
            buffer[q][k] = 0.f;
        }
        v_render_a[q] = v_render_alphas[pix_id[q]];
    }

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = (tile_id == tile_width * tile_height - 1)
                            ? (int32_t)n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = 128;
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Shared memory for batch loading
    extern __shared__ char s[];
    int32_t *id_batch = (int32_t *)s;
    // xy(2) + opacity(1) = 3 floats per entry
    float *xy_opac_batch = (float *)&id_batch[block_size];
    // conic(3) per entry
    float *conic_batch = (float *)&xy_opac_batch[block_size * 3];
    // colors(CDIM) per entry
    float *rgbs_batch = (float *)&conic_batch[block_size * 3];

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int32_t warp_bin_final = 0;
    #pragma unroll
    for(uint32_t q = 0; q < PX; ++q)
        warp_bin_final = max(warp_bin_final, bin_final[q]);
    warp_bin_final = cg::reduce(warp, warp_bin_final, cg::greater<int>());

    for(uint32_t b = 0; b < num_batches; ++b) {
        block.sync();
        const int32_t batch_end = range_end - 1 - (int32_t)(block_size * b);
        const int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - (int32_t)tr;
        if(idx >= range_start) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            xy_opac_batch[tr * 3] = means2d_flat[g * 2];
            xy_opac_batch[tr * 3 + 1] = means2d_flat[g * 2 + 1];
            xy_opac_batch[tr * 3 + 2] = opacities[g];
            conic_batch[tr * 3] = conics_flat[g * 3];
            conic_batch[tr * 3 + 1] = conics_flat[g * 3 + 1];
            conic_batch[tr * 3 + 2] = conics_flat[g * 3 + 2];
            #pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
                rgbs_batch[tr * CDIM + k] = colors_flat[(int64_t)g * CDIM + k];
        }
        block.sync();

        for(uint32_t t = (uint32_t)max(0, batch_end - warp_bin_final); t < (uint32_t)batch_size; ++t) {
            float v_rgb_local[CDIM] = {0.f};
            float v_conic_local[3] = {0.f, 0.f, 0.f};
            float v_xy_local[2] = {0.f, 0.f};
            float v_opacity_local = 0.f;
            bool any_valid = false;

            for(uint32_t q = 0; q < PX; ++q) {
                bool valid_q = inside[q];
                if(batch_end - (int32_t)t > bin_final[q]) valid_q = false;
                if(!valid_q) continue;

                const float cx = conic_batch[t * 3];
                const float cy = conic_batch[t * 3 + 1];
                const float cz = conic_batch[t * 3 + 2];
                const float ox = xy_opac_batch[t * 3];
                const float oy = xy_opac_batch[t * 3 + 1];
                const float opac = xy_opac_batch[t * 3 + 2];
                const float dx = ox - px[q];
                const float dy = oy - py[q];

                // Gaussian weight evaluation (same as eval_gaussian_weight)
                const float power = -0.5f * (cx * dx * dx + cz * dy * dy + cy * dx * dy);
                if(power < -10.0f) continue;
                const float vis = __expf(power);
                const float alpha = opac * vis;
                if(alpha < 1e-4f) continue;

                any_valid = true;

                // VJP computation (reproduce the compute load)
                const float T_q = T[q];
                const float ta = T_q * alpha;

                #pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k) {
                    const float c = rgbs_batch[t * CDIM + k];
                    v_rgb_local[k] += ta * v_render_c[q][k];
                    buffer[q][k] += ta * c;
                }

                const float d_power_dx = -0.5f * (2.0f * cx * dx + cy * dy);
                const float d_power_dy = -0.5f * (2.0f * cz * dy + cy * dx);
                const float d_vis_dx = vis * d_power_dx;
                const float d_vis_dy = vis * d_power_dy;

                float v_alpha = 0.f;
                #pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                    v_alpha += (v_render_c[q][k] * (rgbs_batch[t * CDIM + k] - buffer[q][k] / max(T_q, 1e-6f)));
                v_alpha += v_render_a[q] * (1.0f - T_final[q]);

                v_xy_local[0] += T_q * d_vis_dx * opac * v_alpha;
                v_xy_local[1] += T_q * d_vis_dy * opac * v_alpha;

                v_conic_local[0] += -0.5f * dx * dx * ta * v_alpha / max(alpha, 1e-6f);
                v_conic_local[1] += -0.5f * dx * dy * ta * v_alpha / max(alpha, 1e-6f);
                v_conic_local[2] += -0.5f * dy * dy * ta * v_alpha / max(alpha, 1e-6f);

                v_opacity_local += vis * v_alpha * T_q;

                T[q] = T_q * (1.0f - alpha);
            }

            if(!warp.any(any_valid)) continue;

            // Warp reduction (shuffle-based, same as warpSum)
            for(int offset = 16; offset > 0; offset >>= 1) {
                #pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                    v_rgb_local[k] += __shfl_xor_sync(0xffffffff, v_rgb_local[k], offset);
                v_conic_local[0] += __shfl_xor_sync(0xffffffff, v_conic_local[0], offset);
                v_conic_local[1] += __shfl_xor_sync(0xffffffff, v_conic_local[1], offset);
                v_conic_local[2] += __shfl_xor_sync(0xffffffff, v_conic_local[2], offset);
                v_xy_local[0] += __shfl_xor_sync(0xffffffff, v_xy_local[0], offset);
                v_xy_local[1] += __shfl_xor_sync(0xffffffff, v_xy_local[1], offset);
                v_opacity_local += __shfl_xor_sync(0xffffffff, v_opacity_local, offset);
            }

            if(warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                if(USE_ATOMIC) {
                    // Variant A: atomicAdd (same as real kernel)
                    #pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                        atomicAdd(v_colors_out + (int64_t)CDIM * g + k, v_rgb_local[k]);
                    atomicAdd(v_conics_out + 3 * (int64_t)g, v_conic_local[0]);
                    atomicAdd(v_conics_out + 3 * (int64_t)g + 1, v_conic_local[1]);
                    atomicAdd(v_conics_out + 3 * (int64_t)g + 2, v_conic_local[2]);
                    atomicAdd(v_means2d_out + 2 * (int64_t)g, v_xy_local[0]);
                    atomicAdd(v_means2d_out + 2 * (int64_t)g + 1, v_xy_local[1]);
                    atomicAdd(v_opacities_out + g, v_opacity_local);
                } else {
                    // Variant B: uncontended scratch write
                    const int64_t isect_global = (int64_t)range_start + (int64_t)(b * block_size) + t;
                    const int warp_id = tr / 32;
                    const int64_t si = (isect_global * 4 + warp_id) * 9;
                    #pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k) scratch[si + k] = v_rgb_local[k];
                    scratch[si + 3] = v_conic_local[0];
                    scratch[si + 4] = v_conic_local[1];
                    scratch[si + 5] = v_conic_local[2];
                    scratch[si + 6] = v_xy_local[0];
                    scratch[si + 7] = v_xy_local[1];
                    scratch[si + 8] = v_opacity_local;
                }
            }
        }
    }
}

void run_oracle(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, torch::Tensor tile_offsets, torch::Tensor flatten_ids,
    torch::Tensor render_alphas, torch::Tensor last_ids,
    torch::Tensor v_render_colors, torch::Tensor v_render_alphas,
    int64_t width, int64_t height, int64_t tile_size,
    bool use_atomic, torch::Tensor scratch,
    torch::Tensor v_colors_out, torch::Tensor v_conics_out,
    torch::Tensor v_means2d_out, torch::Tensor v_opacities_out)
{
    constexpr int CDIM = 3;
    const uint32_t N = means2d.size(0);
    const uint32_t n_isects = flatten_ids.size(0);
    const uint32_t tile_w = tile_offsets.size(1);
    const uint32_t tile_h = tile_offsets.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(1, tile_h, tile_w);
    dim3 threads(16, 8, 1);
    // shmem: id_batch(128*4) + xy_opac(128*3*4) + conic(128*3*4) + rgbs(128*3*4)
    const int64_t shmem = 128 * (4 + 12 + 12 + 12);

    auto m2d = means2d.const_data_ptr<float>();
    auto con = conics.const_data_ptr<float>();
    auto col = colors.const_data_ptr<float>();
    auto opa = opacities.const_data_ptr<float>();
    auto toff = tile_offsets.const_data_ptr<int32_t>();
    auto fid = flatten_ids.const_data_ptr<int32_t>();
    auto ra = render_alphas.const_data_ptr<float>();
    auto li = last_ids.const_data_ptr<int32_t>();
    auto vrc = v_render_colors.const_data_ptr<float>();
    auto vra = v_render_alphas.const_data_ptr<float>();
    auto scr = scratch.data_ptr<float>();
    auto vco = v_colors_out.data_ptr<float>();
    auto vcon = v_conics_out.data_ptr<float>();
    auto vm2 = v_means2d_out.data_ptr<float>();
    auto vop = v_opacities_out.data_ptr<float>();

    if(use_atomic) {
        cudaFuncSetAttribute(scatter_oracle_kernel<3,2,true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        scatter_oracle_kernel<3,2,true><<<grid, threads, (size_t)shmem, stream>>>(
            n_isects, m2d, con, col, opa, tile_w, tile_h,
            (uint32_t)width, (uint32_t)height, (uint32_t)tile_size,
            toff, fid, ra, li, vrc, vra, vco, vcon, vm2, vop, scr);
    } else {
        cudaFuncSetAttribute(scatter_oracle_kernel<3,2,false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        scatter_oracle_kernel<3,2,false><<<grid, threads, (size_t)shmem, stream>>>(
            n_isects, m2d, con, col, opa, tile_w, tile_h,
            (uint32_t)width, (uint32_t)height, (uint32_t)tile_size,
            toff, fid, ra, li, vrc, vra, vco, vcon, vm2, vop, scr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_oracle", &run_oracle, "Atomic-free oracle microbenchmark");
}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="room")
    ap.add_argument("--cam-idx", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-long-side", type=int, default=2048)
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-measure", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = f"cuda:{args.gpu}"

    from plyfile import PlyData
    SH_DEGREE = 3
    K_SH = (SH_DEGREE + 1) ** 2

    scene_configs = {
        "room": {
            "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
            "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
            "native_w": 3114, "native_h": 2075,
        },
    }
    cfg = scene_configs[args.scene]
    nw, nh = cfg["native_w"], cfg["native_h"]
    if args.max_long_side > 0 and max(nw, nh) > args.max_long_side:
        scale = args.max_long_side / max(nw, nh)
        width, height = max(1, int(round(nw * scale))), max(1, int(round(nh * scale)))
    else:
        width, height = nw, nh

    ply = PlyData.read(cfg["ply"])
    v = ply["vertex"]
    N = len(v)
    means = torch.tensor(np.column_stack([v["x"], v["y"], v["z"]]), dtype=torch.float32, device=device)
    quats = torch.tensor(np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]), dtype=torch.float32, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.exp(torch.tensor(np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]), dtype=torch.float32, device=device))
    opacities = torch.sigmoid(torch.tensor(v["opacity"], dtype=torch.float32, device=device))
    f_dc = torch.tensor(np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]), dtype=torch.float32, device=device)
    n_rest = 3 * (K_SH - 1)
    f_rest = torch.stack([torch.tensor(v[f"f_rest_{i}"], dtype=torch.float32, device=device) for i in range(n_rest)], dim=1)
    f_rest = f_rest.reshape(N, 3, K_SH - 1).permute(0, 2, 1)
    sh = torch.zeros(N, K_SH, 3, dtype=torch.float32, device=device)
    sh[:, 0] = f_dc
    sh[:, 1:] = f_rest
    colors = sh

    with open(cfg["cams"]) as f:
        cams = json_mod = json
        cams = json.load(f)
    c = cams[args.cam_idx]
    R = np.asarray(c["rotation"], dtype=np.float64)
    p = np.asarray(c["position"], dtype=np.float64)
    Rw2c = R.T
    vm = np.eye(4)
    vm[:3, :3] = Rw2c
    vm[:3, 3] = -Rw2c @ p
    scale = width / float(c["width"])
    K = np.array([[float(c["fx"]) * scale, 0.0, (width - 1) / 2.0],
                  [0.0, float(c["fy"]) * scale, (height - 1) / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    vm_t = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    K_t = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    tile_size = 16
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)

    from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode, _make_lazy_cuda_func
    from gsplat.rendering import _maybe_evaluate_sh
    from gsplat.experimental.render.functional.gaussian_inference import (
        _cull_gaussians_batched, _gather_visible_native,
    )

    with torch.no_grad():
        visible_ids, _, _ = _cull_gaussians_batched(
            means, quats, scales, vm_t, K_t, width, height, eps2d=0.3,
            near_plane=0.01, far_plane=1e10, radius_clip=0.0, camera_model="pinhole",
        )
        v_means, v_quats, v_scales, v_opacities, v_colors = _gather_visible_native(
            means, quats, scales, opacities, colors, visible_ids,
        )
        N_visible = visible_ids.numel()

        v_means_b = v_means.unsqueeze(0).contiguous()
        v_quats_b = v_quats.unsqueeze(0).contiguous()
        v_scales_b = v_scales.unsqueeze(0).contiguous()
        v_opacities_b = v_opacities.unsqueeze(0).contiguous()
        v_colors_input = v_colors.unsqueeze(0) if v_colors.dim() == 2 else v_colors
        opacities_bc = torch.broadcast_to(v_opacities_b[..., None, :], (1, 1, N_visible)).contiguous()

        radii, means2d, depths, conics, _ = fully_fused_projection(
            means=v_means_b, covars=None, quats=v_quats_b, scales=v_scales_b,
            viewmats=vm_t, Ks=K_t, width=width, height=height, eps2d=0.3,
            near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False,
            calc_compensations=False, camera_model="pinhole",
        )
        colors_eval = _maybe_evaluate_sh(
            SH_DEGREE, v_colors_input, v_means_b, radii, vm_t, (1,), 1, N_visible, True,
        ).contiguous()

        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height,
            packed=False, n_images=1, image_ids=None, gaussian_ids=None,
            conics=conics, opacities=opacities_bc,
        )
        isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height).reshape((1, 1, tile_height, tile_width))

        render_out = _make_lazy_cuda_func("rasterize_to_pixels_3dgs")(
            means2d.contiguous(), conics.contiguous(),
            colors_eval.contiguous(), opacities_bc.contiguous(),
            None, None, width, height, tile_size,
            isect_offsets.contiguous(), flatten_ids.contiguous(),
            False, False,
        )
        render_colors, render_alphas, _, last_ids = render_out

    n_isects = isect_ids.numel()
    print(f"[oracle] N_visible={N_visible}, n_isects={n_isects}, tiles={tile_width}x{tile_height}")

    # Prepare flat inputs
    means2d_flat = means2d[0].contiguous()    # [N_v, 2]
    conics_flat = conics[0].contiguous()      # [N_v, 3]
    colors_flat = colors_eval[0].contiguous() # [N_v, 3]
    opacities_flat = v_opacities.contiguous() # [N_v]
    tile_offsets_flat = isect_offsets[0, 0].contiguous()  # [tile_h, tile_w]
    flatten_ids_flat = flatten_ids.contiguous().to(torch.int32)
    render_alphas_flat = render_alphas[0].squeeze(-1).contiguous()  # [H, W]
    last_ids_flat = last_ids[0].contiguous().to(torch.int32)       # [H, W]
    v_render_colors = torch.rand(1, height, width, 3, dtype=torch.float32, device=device) * 0.01
    v_render_alphas = torch.rand(1, height, width, dtype=torch.float32, device=device) * 0.01
    v_render_colors_flat = v_render_colors[0].contiguous()
    v_render_alphas_flat = v_render_alphas[0].contiguous()

    # Output buffers (zeroed each iteration for atomic variant)
    v_colors_out = torch.zeros(N_visible, 3, dtype=torch.float32, device=device)
    v_conics_out = torch.zeros(N_visible, 3, dtype=torch.float32, device=device)
    v_means2d_out = torch.zeros(N_visible, 2, dtype=torch.float32, device=device)
    v_opacities_out = torch.zeros(N_visible, dtype=torch.float32, device=device)
    scratch = torch.zeros(n_isects * 4 * 9, dtype=torch.float32, device=device)

    # Build extension
    print("[oracle] Compiling CUDA extension...")
    from torch.utils.cpp_extension import load_inline
    ext = load_inline(
        name="h2_bwd_0_oracle",
        cpp_sources=[],
        cuda_sources=[CUDA_SOURCE],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=True,
    )
    print("[oracle] Compilation done.")

    def run_variant(use_atomic):
        if use_atomic:
            v_colors_out.zero_()
            v_conics_out.zero_()
            v_means2d_out.zero_()
            v_opacities_out.zero_()
        ext.run_oracle(
            means2d_flat, conics_flat, colors_flat, opacities_flat,
            tile_offsets_flat, flatten_ids_flat,
            render_alphas_flat, last_ids_flat,
            v_render_colors_flat, v_render_alphas_flat,
            width, height, tile_size, use_atomic, scratch,
            v_colors_out, v_conics_out, v_means2d_out, v_opacities_out
        )

    def time_variant(use_atomic, n_warmup, n_measure):
        for _ in range(n_warmup):
            run_variant(use_atomic)
        torch.cuda.synchronize(device)
        times = []
        for _ in range(n_measure):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_variant(use_atomic)
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
        return np.array(times)

    print("[oracle] Timing atomic variant (A)...")
    times_atomic = time_variant(True, args.n_warmup, args.n_measure)
    print(f"  T_atomic: median={np.median(times_atomic):.3f}ms mean={np.mean(times_atomic):.3f}ms")

    print("[oracle] Timing atomic-free variant (B)...")
    times_free = time_variant(False, args.n_warmup, args.n_measure)
    print(f"  T_free: median={np.median(times_free):.3f}ms mean={np.mean(times_free):.3f}ms")

    T_blend_current = float(np.median(times_atomic))
    T_blend_atomic_free = float(np.median(times_free))
    O_atomic_max = T_blend_current - T_blend_atomic_free

    # Also measure real F+B for reference
    print("[oracle] Measuring real F+B for reference...")
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import (
        create_higs_renderer, _HIGS_FROZEN_TRACKER,
    )

    _HIGS_FROZEN_TRACKER.reset()
    handle = create_higs_renderer(means.detach(), quats.detach(), scales.detach(),
                                  opacities.detach(), colors.detach(), sh_degree=SH_DEGREE)
    torch.cuda.synchronize()

    # Reference render
    with torch.no_grad():
        ref_res = rasterize_gaussian_higs_frozen(
            means.detach(), quats.detach(), scales.detach(), opacities.detach(), colors.detach(),
            backward_mode="higs_native", scene=handle, freeze_topology=True,
            viewmats=vm_t, Ks=K_t, width=width, height=height,
            sh_degree=SH_DEGREE, use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0,
        )
    target = ref_res["frame"].detach().clone()

    def fb_fn():
        m = means.detach().clone().requires_grad_(True)
        q = quats.detach().clone().requires_grad_(True)
        s = scales.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        c = colors.detach().clone().requires_grad_(True)
        res = rasterize_gaussian_higs_frozen(
            m, q, s, o, c, backward_mode="higs_native", scene=handle, freeze_topology=True,
            viewmats=vm_t, Ks=K_t, width=width, height=height,
            sh_degree=SH_DEGREE, use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0,
        )
        loss = (res["frame"] - target).abs().mean()
        loss.backward()

    real_times = []
    for _ in range(args.n_warmup):
        fb_fn()
    torch.cuda.synchronize(device)
    for _ in range(args.n_measure):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fb_fn()
        end.record()
        torch.cuda.synchronize(device)
        real_times.append(start.elapsed_time(end))

    real_times = np.array(real_times)
    T_fb_real = float(np.median(real_times))
    print(f"  T_F+B_real: median={T_fb_real:.3f}ms")

    results = {
        "scene": args.scene, "camera_idx": args.cam_idx,
        "width": width, "height": height,
        "N_visible": int(N_visible), "n_isects": int(n_isects),
        "n_warmup": args.n_warmup, "n_measure": args.n_measure,
        "T_blend_current_ms": T_blend_current,
        "T_blend_atomic_free_ms": T_blend_atomic_free,
        "O_atomic_max_ms": O_atomic_max,
        "O_atomic_max_pct_blend": O_atomic_max / T_blend_current if T_blend_current > 0 else 0,
        "O_atomic_max_pct_fb": O_atomic_max / T_fb_real if T_fb_real > 0 else 0,
        "T_fb_real_ms": T_fb_real,
        "times_atomic_ms": [float(t) for t in times_atomic],
        "times_free_ms": [float(t) for t in times_free],
        "times_real_fb_ms": [float(t) for t in real_times],
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n===== ATOMIC-FREE ORACLE RESULTS =====")
    print(f"T_blend_current (atomic):  {T_blend_current:.3f} ms")
    print(f"T_blend_atomic_free:       {T_blend_atomic_free:.3f} ms")
    print(f"O_atomic_max:              {O_atomic_max:.3f} ms")
    print(f"O_atomic_max / T_blend:    {results['O_atomic_max_pct_blend']*100:.1f}%")
    print(f"O_atomic_max / T_F+B:      {results['O_atomic_max_pct_fb']*100:.1f}%")
    print(f"T_F+B_real (reference):    {T_fb_real:.3f} ms")
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
