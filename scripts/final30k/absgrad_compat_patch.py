#!/usr/bin/env python3
"""FINAL30K absgrad compatibility patch (C0_V3_FINAL30K).

Creates /mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree as an exact
copy of the frozen research worktree (higs_c0_worktree -> C0_V3_RESEARCH_FROZEN
sources) and applies the MINIMAL patch that adds the auxiliary absolute
view-space gradient (absgrad) densification statistic required by the frozen
FINAL-30K protocol (densification signal = absgrad, threshold 0.0008).

Scope (BENCHMARK_PROTOCOL_COMPATIBILITY, not an optimization):
  * RasterizeToPixels3DGSDevice.cuh: under compute_abs && H8_MR, compute the
    conic-CONTRACTED per-pixel quantity (reference gsplat semantics) instead
    of abs() of the moment-space v_xy_local. Non-H8_MR paths unchanged.
  * HigsNativeBackward.cu: new v_means2d_abs output buffer on both blend-bwd
    kernels (nullptr disables), warpSum + atomicAdd scatter mirroring the
    gsplat reference kernel exactly; wrapper allocates the buffer, reads
    HIGS_BWD_ABSGRAD=1, returns it as an 8th tuple element.
  * HigsNativeBackward.h: 8-tuple return type.
  * gaussian_inference.py: unpack the 8th element; when HIGS_BWD_ABSGRAD=1,
    scatter to [C, N, 2] and attach as means2d_proxy.absgrad (mirrors gsplat).

Unchanged: F9 producer, SCALAR_ADJOINT, H8-MR signed paths, forward raster,
visibility/support/intersection semantics, RGB/alpha, ALL signed gradients
(means/quats/scales/opacity/SH/means2d), H8 moments, optimizer-facing
gradients, loss, optimizer, thresholds, schedules.
"""
import json
import os
import shutil
import subprocess
import sys

SRC_WORKTREE = "/mnt/storage_pool/liaoyuanjun/higs_c0_worktree"
DST_WORKTREE = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree"

CU = "gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu"
H = "gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.h"
CUH = "gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh"
PY = "gsplat/experimental/render/functional/gaussian_inference.py"

edits = []  # (file, anchor_id, count, ok)


def patch(path, old, new, expect=1, anchor=""):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    n = text.count(old)
    if n != expect:
        print(f"ANCHOR FAIL [{anchor}] in {path}: found {n}, expected {expect}")
        sys.exit(2)
    text = text.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    edits.append({"file": path, "anchor": anchor, "replacements": n})
    print(f"  patched [{anchor}] x{n}")


def main():
    if os.path.exists(DST_WORKTREE):
        print(f"REFUSING: {DST_WORKTREE} already exists; move it away first")
        sys.exit(3)
    print(f"copying {SRC_WORKTREE} -> {DST_WORKTREE} ...")
    shutil.copytree(SRC_WORKTREE, DST_WORKTREE)
    cu = os.path.join(DST_WORKTREE, CU)
    h = os.path.join(DST_WORKTREE, H)
    cuh = os.path.join(DST_WORKTREE, CUH)
    py = os.path.join(DST_WORKTREE, PY)

    # ---------------- device function: H8-MR-contracted absgrad -------------
    patch(
        cuh,
        """        if(compute_abs)
        {
            v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
        }""",
        """        if(compute_abs)
        {
            if constexpr(H8_MR)
            {
                // FINAL30K absgrad compatibility: under H8-MR the signed
                // v_xy_local is a moment-space quantity (the conic contraction
                // is deferred to the projection backward). The reference
                // absgrad (gsplat absgrad=True) is the absolute value of the
                // CONTRACTED per-pixel view-space contribution, taken per
                // (pixel, gaussian) pair BEFORE any reduction, so recompute
                // the contraction here with the same v_sigma expression as the
                // reference kernel. Signed outputs are untouched.
                const float v_sigma_ref = -opac * vis * v_alpha;
                v_xy_abs_local = {
                    fabsf(v_sigma_ref * (conic.x * delta.x + conic.y * delta.y)),
                    fabsf(v_sigma_ref * (conic.y * delta.x + conic.z * delta.y))
                };
            }
            else
            {
                v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
            }
        }""",
        anchor="device_compute_abs_h8mr",
    )

    # ---------------- main kernel signature --------------------------------
    patch(
        cu,
        """    float *__restrict__ v_opacities,
    float *__restrict__ v_backgrounds // [CDIM] (summed over all images)
)
{
    auto block = cg::this_thread_block();""",
        """    float *__restrict__ v_opacities,
    float *__restrict__ v_backgrounds, // [CDIM] (summed over all images)
    vec2 *__restrict__ v_means2d_abs   // [I * N, 2] FINAL30K absgrad (auxiliary; nullptr disables)
)
{
    auto block = cg::this_thread_block();""",
        anchor="main_kernel_sig",
    )

    # ---------------- main kernel per-t locals -----------------------------
    patch(
        cu,
        """            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local      = {0.f, 0.f, 0.f};
            vec2 v_xy_local         = {0.f, 0.f};
            float v_opacity_local   = 0.f;
            // initialize everything to 0, only set if the lane is valid""",
        """            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local      = {0.f, 0.f, 0.f};
            vec2 v_xy_local         = {0.f, 0.f};
            vec2 v_xy_abs_local     = {0.f, 0.f};
            float v_opacity_local   = 0.f;
            // initialize everything to 0, only set if the lane is valid""",
        anchor="main_kernel_locals",
    )

    # ---------------- main kernel call site --------------------------------
    patch(
        cu,
        """                    false, // compute_abs
                    T,
                    buffer,
                    v_rgb_local,
                    v_conic_local,
                    v_xy_local,
                    v_xy_local, // v_xy_abs_local (unused when compute_abs=false)
                    v_opacity_local
                );""",
        """                    (v_means2d_abs != nullptr), // compute_abs (FINAL30K absgrad compatibility)
                    T,
                    buffer,
                    v_rgb_local,
                    v_conic_local,
                    v_xy_local,
                    v_xy_abs_local,
                    v_opacity_local
                );""",
        anchor="main_kernel_call",
    )

    # ---------------- main kernel warp reduction ---------------------------
    patch(
        cu,
        """            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            if(warp.thread_rank() == 0)
            {
                int32_t g        = id_batch[t]; // flatten index in [I * N]""",
        """            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            if(v_means2d_abs != nullptr)
            {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
            if(warp.thread_rank() == 0)
            {
                int32_t g        = id_batch[t]; // flatten index in [I * N]""",
        anchor="main_kernel_warpsum",
    )

    # ---------------- main kernel atomic scatter ---------------------------
    patch(
        cu,
        """                float *v_xy_ptr = (float *)(v_means2d) + 2 * (int64_t)g;
                atomicAdd(v_xy_ptr, v_xy_local.x);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y);

                atomicAdd(v_opacities + g, v_opacity_local);""",
        """                float *v_xy_ptr = (float *)(v_means2d) + 2 * (int64_t)g;
                atomicAdd(v_xy_ptr, v_xy_local.x);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if(v_means2d_abs != nullptr)
                {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * (int64_t)g;
                    atomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    atomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                atomicAdd(v_opacities + g, v_opacity_local);""",
        anchor="main_kernel_scatter",
    )

    # ---------------- px kernel signature ----------------------------------
    patch(
        cu,
        """    float *__restrict__ v_opacities,
    float *__restrict__ v_backgrounds
)
{
    static_assert(PX == 1 || PX == 2 || PX == 4 || PX == 8, "PX must be 1/2/4/8");""",
        """    float *__restrict__ v_opacities,
    float *__restrict__ v_backgrounds,
    vec2 *__restrict__ v_means2d_abs // [I * N, 2] FINAL30K absgrad (auxiliary; nullptr disables)
)
{
    static_assert(PX == 1 || PX == 2 || PX == 4 || PX == 8, "PX must be 1/2/4/8");""",
        anchor="px_kernel_sig",
    )

    # ---------------- px kernel per-t locals -------------------------------
    patch(
        cu,
        """            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local      = {0.f, 0.f, 0.f};
            vec2 v_xy_local         = {0.f, 0.f};
            float v_opacity_local   = 0.f;
            bool any_valid          = false;""",
        """            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local      = {0.f, 0.f, 0.f};
            vec2 v_xy_local         = {0.f, 0.f};
            vec2 v_xy_abs_local     = {0.f, 0.f};
            float v_opacity_local   = 0.f;
            bool any_valid          = false;""",
        anchor="px_kernel_locals",
    )

    # ---------------- px kernel per-q scratch ------------------------------
    patch(
        cu,
        """                        float v_rgb_q[CDIM] = {0.f};
                        vec3 v_conic_q      = {0.f, 0.f, 0.f};
                        vec2 v_xy_q         = {0.f, 0.f};
                        float v_opacity_q   = 0.f;""",
        """                        float v_rgb_q[CDIM] = {0.f};
                        vec3 v_conic_q      = {0.f, 0.f, 0.f};
                        vec2 v_xy_q         = {0.f, 0.f};
                        vec2 v_xy_abs_q     = {0.f, 0.f};
                        float v_opacity_q   = 0.f;""",
        anchor="px_kernel_q_scratch",
    )

    # ---------------- px kernel call site ----------------------------------
    patch(
        cu,
        """                            false, // compute_abs
                            T[q],
                            buffer[q],
                            v_rgb_q,
                            v_conic_q,
                            v_xy_q,
                            v_xy_q,
                            v_opacity_q
                        );""",
        """                            (v_means2d_abs != nullptr), // compute_abs (FINAL30K absgrad compatibility)
                            T[q],
                            buffer[q],
                            v_rgb_q,
                            v_conic_q,
                            v_xy_q,
                            v_xy_abs_q,
                            v_opacity_q
                        );""",
        anchor="px_kernel_call",
    )

    # ---------------- px kernel per-q accumulation -------------------------
    patch(
        cu,
        """                        v_xy_local.x += v_xy_q.x;
                        v_xy_local.y += v_xy_q.y;
                        v_opacity_local += v_opacity_q;""",
        """                        v_xy_local.x += v_xy_q.x;
                        v_xy_local.y += v_xy_q.y;
                        v_xy_abs_local.x += v_xy_abs_q.x;
                        v_xy_abs_local.y += v_xy_abs_q.y;
                        v_opacity_local += v_opacity_q;""",
        anchor="px_kernel_accum",
    )

    # ---------------- px kernel warp reduction -----------------------------
    patch(
        cu,
        """            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            if(warp.thread_rank() == 0)
            {
                int32_t g        = id_batch[t];""",
        """            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            if(v_means2d_abs != nullptr)
            {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
            if(warp.thread_rank() == 0)
            {
                int32_t g        = id_batch[t];""",
        anchor="px_kernel_warpsum",
    )

    # ---------------- px kernel atomic scatter -----------------------------
    patch(
        cu,
        """                float *v_xy_ptr = (float *)(v_means2d) + 2 * (int64_t)g;
                atomicAdd(v_xy_ptr, v_xy_local.x);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y);
                atomicAdd(v_opacities + g, v_opacity_local);""",
        """                float *v_xy_ptr = (float *)(v_means2d) + 2 * (int64_t)g;
                atomicAdd(v_xy_ptr, v_xy_local.x);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y);
                if(v_means2d_abs != nullptr)
                {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * (int64_t)g;
                    atomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    atomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }
                atomicAdd(v_opacities + g, v_opacity_local);""",
        anchor="px_kernel_scatter",
    )

    # ---------------- wrapper: buffer allocation ---------------------------
    patch(
        cu,
        """    at::Tensor v_backgrounds = at::zeros({color_dim}, opts);
    at::Tensor v_means2d     = at::zeros({I * N, 2}, opts);""",
        """    at::Tensor v_backgrounds = at::zeros({color_dim}, opts);
    at::Tensor v_means2d     = at::zeros({I * N, 2}, opts);
    // FINAL30K absgrad compatibility: auxiliary absolute view-space gradient
    // (densification statistic ONLY). Allocated unconditionally so the return
    // contract is stable; written by the blend backward only when
    // HIGS_BWD_ABSGRAD=1. Zero tensor otherwise. No signed gradient or
    // rendering output depends on this buffer.
    at::Tensor v_means2d_abs = at::zeros({I * N, 2}, opts);""",
        anchor="wrapper_alloc",
    )

    # ---------------- wrapper: env toggle ----------------------------------
    patch(
        cu,
        """    const char *h8_mr_env = std::getenv("HIGS_BWD_H8_MR");
    const bool h8_mr = (h8_mr_env != nullptr) && (strcmp(h8_mr_env, "1") == 0);""",
        """    const char *h8_mr_env = std::getenv("HIGS_BWD_H8_MR");
    const bool h8_mr = (h8_mr_env != nullptr) && (strcmp(h8_mr_env, "1") == 0);
    // FINAL30K absgrad compatibility: enable the auxiliary absgrad statistic.
    // Off by default. Changes NO signed gradient and NO rendering output.
    const char *absgrad_env = std::getenv("HIGS_BWD_ABSGRAD");
    const bool absgrad_enabled =
        (absgrad_env != nullptr) && (strcmp(absgrad_env, "1") == 0);
    vec2 *v_means2d_abs_ptr =
        absgrad_enabled ? reinterpret_cast<vec2 *>(v_means2d_abs.data_ptr<float>()) : nullptr;""",
        anchor="wrapper_env",
    )

    # ---------------- wrapper: launcher macro arg (5 branches) -------------
    patch(
        cu,
        "v_backgrounds.data_ptr<float>());",
        "v_backgrounds.data_ptr<float>(), v_means2d_abs_ptr);",
        expect=5,
        anchor="launcher_macro_args",
    )

    # ---------------- wrapper: return type (.cu) ---------------------------
    patch(
        cu,
        """std::tuple<
    at::Tensor, // v_means [N, 3]
    at::Tensor, // v_quats [N, 4]
    at::Tensor, // v_scales [N, 3]
    at::Tensor, // v_opacities [N]
    at::Tensor, // v_colors_master
    at::Tensor, // v_backgrounds [3]
    at::Tensor  // v_means2d [I*N, 2]
    >
higs_rasterize_backward(""",
        """std::tuple<
    at::Tensor, // v_means [N, 3]
    at::Tensor, // v_quats [N, 4]
    at::Tensor, // v_scales [N, 3]
    at::Tensor, // v_opacities [N]
    at::Tensor, // v_colors_master
    at::Tensor, // v_backgrounds [3]
    at::Tensor, // v_means2d [I*N, 2]
    at::Tensor  // v_means2d_abs [I*N, 2] (FINAL30K absgrad; zeros unless HIGS_BWD_ABSGRAD=1)
    >
higs_rasterize_backward(""",
        anchor="wrapper_ret_type_cu",
    )

    # ---------------- wrapper: return value --------------------------------
    patch(
        cu,
        """    return {
        grad_means,
        grad_quats,
        grad_scales,
        grad_opacities,
        grad_colors,
        v_backgrounds,
        v_means2d
    };""",
        """    return {
        grad_means,
        grad_quats,
        grad_scales,
        grad_opacities,
        grad_colors,
        v_backgrounds,
        v_means2d,
        v_means2d_abs
    };""",
        anchor="wrapper_ret_value",
    )

    # ---------------- header: return type ----------------------------------
    patch(
        h,
        """    std::tuple<
        at::Tensor, // v_means    [N, 3]
        at::Tensor, // v_quats    [N, 4]
        at::Tensor, // v_scales   [N, 3]
        at::Tensor, // v_opacities [N]
        at::Tensor, // v_colors_master
        at::Tensor, // v_backgrounds [3]
        at::Tensor  // v_means2d [I*N, 2]
        >""",
        """    std::tuple<
        at::Tensor, // v_means    [N, 3]
        at::Tensor, // v_quats    [N, 4]
        at::Tensor, // v_scales   [N, 3]
        at::Tensor, // v_opacities [N]
        at::Tensor, // v_colors_master
        at::Tensor, // v_backgrounds [3]
        at::Tensor, // v_means2d [I*N, 2]
        at::Tensor  // v_means2d_abs [I*N, 2] (FINAL30K absgrad; zeros unless HIGS_BWD_ABSGRAD=1)
        >""",
        anchor="header_ret_type",
    )

    # ---------------- python glue: unpack 8-tuple --------------------------
    patch(
        py,
        """        (
            _v_means_g, _v_quats_g, _v_scales_g, _v_opacities_g,
            _v_colors_master_g, v_backgrounds_g, v_means2d_g,
        ) = out""",
        """        (
            _v_means_g, _v_quats_g, _v_scales_g, _v_opacities_g,
            _v_colors_master_g, v_backgrounds_g, v_means2d_g,
            v_means2d_abs_g,
        ) = out""",
        anchor="py_unpack",
    )

    # ---------------- python glue: attach absgrad --------------------------
    patch(
        py,
        """        grad_means2d = torch.zeros_like(means2d_proxy)
        grad_means2d.index_copy_(
            1, visible_ids, v_means2d_g.reshape(I, ctx.N_visible, 2)
        )""",
        """        grad_means2d = torch.zeros_like(means2d_proxy)
        grad_means2d.index_copy_(
            1, visible_ids, v_means2d_g.reshape(I, ctx.N_visible, 2)
        )

        if os.environ.get("HIGS_BWD_ABSGRAD", "0") == "1":
            # FINAL30K absgrad compatibility: attach the reference-semantics
            # absolute view-space gradient to the means2d proxy, mirroring
            # gsplat's ``means2d.absgrad = v_means2d_abs``. Invisible rows
            # stay zero, matching gsplat's zeros_like allocation. Consumed by
            # add_densification_stats (absgrad preference). This attribute is
            # an auxiliary densification statistic only: no signed gradient
            # path reads it.
            grad_means2d_abs = torch.zeros_like(means2d_proxy)
            grad_means2d_abs.index_copy_(
                1, visible_ids, v_means2d_abs_g.reshape(I, ctx.N_visible, 2)
            )
            means2d_proxy.absgrad = grad_means2d_abs""",
        anchor="py_attach",
    )

    # ---------------- verification -----------------------------------------
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", py], capture_output=True, text=True
    )
    if r.returncode != 0:
        print("PY_COMPILE FAIL:\n" + r.stderr)
        sys.exit(4)
    print("py_compile OK")

    # record per-file sha256 + diff stats vs source worktree
    import hashlib

    report = {"edits": edits, "files": {}}
    for rel in (CU, H, CUH, PY):
        sp = os.path.join(SRC_WORKTREE, rel)
        dp = os.path.join(DST_WORKTREE, rel)
        with open(sp, "rb") as f:
            src_sha = hashlib.sha256(f.read()).hexdigest()
        with open(dp, "rb") as f:
            dst_sha = hashlib.sha256(f.read()).hexdigest()
        diff = subprocess.run(
            ["diff", "-u", sp, dp], capture_output=True, text=True
        ).stdout
        added = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
        report["files"][rel] = {
            "src_sha256": src_sha,
            "dst_sha256": dst_sha,
            "lines_added": added,
            "lines_removed": removed,
        }
        print(f"{rel}: +{added}/-{removed}  {src_sha[:12]} -> {dst_sha[:12]}")

    out_path = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_patch_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"PATCH COMPLETE: {len(edits)} anchors, report -> {out_path}")


if __name__ == "__main__":
    main()
