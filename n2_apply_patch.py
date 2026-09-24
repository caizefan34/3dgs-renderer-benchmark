#!/usr/bin/env python3
"""Apply CPCB (Cotangent-Projected Compositing Backward) patch to RasterizeToPixels3DGSBwd.cu

This is an EXACT real-valued VJP reformulation that replaces the vector-valued
suffix state buffer[CDIM] with a single scalar suffix_proj.

Math: v_alpha_i = T_i * (dot(color_i, q) + qa - suffix)
       suffix_{i-1} = suffix_i + alpha_i * (dot(color_i, q) + qa - suffix_i)

Only the backward kernel is modified. Forward, sigma, alpha, atomics, shared
memory layout, warp reductions are all unchanged.
"""
import sys
import shutil
from pathlib import Path

def apply_patch(cu_path: str, dry_run: bool = False) -> str:
    p = Path(cu_path)
    original = p.read_text()
    src = original

    # --- Change 1: Replace buffer[CDIM] declaration with suffix_proj ---
    old1 = "    float buffer[CDIM] = {0.f};"
    new1 = "    float suffix_proj = 0.f;  // CPCB: scalar suffix state (replaces buffer[CDIM])"
    assert old1 in src, "Cannot find buffer[CDIM] declaration"
    src = src.replace(old1, new1, 1)

    # --- Change 2: Add background initialization AFTER v_render_c is loaded ---
    # v_render_c loaded at: const float v_render_a = v_render_alphas[pix_id];
    # We insert the background init right after that line.
    old2 = "    const float v_render_a = v_render_alphas[pix_id];\n"
    new2 = (
        "    const float v_render_a = v_render_alphas[pix_id];\n"
        "\n"
        "    // CPCB: initialize suffix projection with background dot product.\n"
        "    // Must occur AFTER v_render_c[] is loaded.\n"
        "    if (backgrounds != nullptr) {\n"
        "#pragma unroll\n"
        "        for (uint32_t k = 0; k < CDIM; ++k) {\n"
        "            suffix_proj += backgrounds[k] * v_render_c[k];\n"
        "        }\n"
        "    }\n"
    )
    assert old2 in src, "Cannot find v_render_a loading line"
    src = src.replace(old2, new2, 1)

    # --- Change 3: Replace v_alpha computation ---
    old3 = """                float v_alpha = 0.f;
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
                }"""
    new3 = """                // CPCB: scalar cotangent-projected v_alpha computation.
                // Replaces vector buffer[CDIM] dot products with single scalar.
                float current_proj = v_render_a;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    current_proj +=
                        rgbs_batch[t * CDIM + k] * v_render_c[k];
                }

                const float proj_diff =
                    current_proj - suffix_proj;

                const float v_alpha =
                    T * proj_diff;"""
    assert old3 in src, "Cannot find v_alpha computation block"
    src = src.replace(old3, new3, 1)

    # --- Change 4: Replace buffer update with suffix_proj update ---
    old4 = """#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }"""
    new4 = """                // CPCB: update suffix projection (replaces buffer[k] += color[k] * fac)
                suffix_proj += alpha * proj_diff;"""
    assert old4 in src, "Cannot find buffer update block"
    src = src.replace(old4, new4, 1)

    # Verify no remaining references to buffer[] (except in comments)
    remaining_buffer = [i for i, line in enumerate(src.split('\n'), 1)
                        if 'buffer' in line and 'CPCB' not in line and '//' not in line.split('buffer')[0]]
    if remaining_buffer:
        print(f"WARNING: remaining buffer references at lines: {remaining_buffer}")
        for ln in remaining_buffer:
            print(f"  line {ln}: {src.split(chr(10))[ln-1]}")

    if dry_run:
        print("DRY RUN - changes verified but not written")
        return src

    # Backup and write
    backup = p.with_suffix('.cu.bak')
    if not backup.exists():
        shutil.copy2(p, backup)
    p.write_text(src)
    print(f"Patch applied to {cu_path}")
    print(f"Backup saved to {backup}")
    return src

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry = "--dry-run" in sys.argv
    cu_path = args[0] if args else "gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu"
    apply_patch(cu_path, dry_run=dry)
