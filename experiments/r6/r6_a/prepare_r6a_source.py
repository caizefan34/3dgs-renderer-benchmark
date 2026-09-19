#!/usr/bin/env python3
"""Create an isolated, R6-A-patched gsplat source tree with debug atomic
instrumentation in the rasterize_to_pixels_3dgs_bwd_kernel.

The patch injects a __device__ counter + host-side reset/read functions
DIRECTLY INTO the kernel .cu file.  This is required because CUDA __device__
variables do not have cross-TU linkage (extern __device__ is treated as static
by nvcc), so a separate .cu file would get a different counter than the kernel.

Supports two gsplat source layouts:

  Modern (>= 1.5.0, e.g. 1.5.3 on A100):
    cuda/csrc/RasterizeToPixels3DGSBwd.cu  — backward kernel
    cuda/ext.cpp                           — pybind11 bindings

  Legacy (< 1.5.0):
    csrc/rasterize_to_pixels_bwd.cu  — backward kernel
    csrc/ext.cpp                     — pybind11 bindings

Usage:
  python prepare_r6a_source.py \
    --source ~/.local/lib/python3.10/site-packages/gsplat \
    --output /tmp/r6a_patched/gsplat
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HEADER = ROOT / "r6a_debug_atomic.cuh"


# Code injected at the TOP of the kernel .cu file, inside namespace gsplat,
# before the kernel template.  Includes the __device__ counter definition
# and host-side reset/read functions.
COUNTER_AND_FUNCS = """
// === R6-A Direct Debug Atomic Instrumentation ===
// Counter incremented once per warp-leader atomicAdd execution set.
// Defined HERE (not extern) because CUDA __device__ variables have no
// cross-TU linkage.
__device__ unsigned long long r6a_debug_count = 0ULL;

namespace r6a {
void reset_debug_counter() {
    unsigned long long zero = 0ULL;
    cudaMemcpyToSymbol(r6a_debug_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
}
uint64_t get_debug_counter() {
    unsigned long long val = 0ULL;
    cudaMemcpyFromSymbol(&val, r6a_debug_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    return (uint64_t)val;
}
} // namespace r6a
// === End R6-A Instrumentation ===

"""


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected R6-A patch context missing: {path}")
    path.write_text(text.replace(old, new, 1))


def detect_layout(source: Path) -> str:
    csrc = source / "cuda" / "csrc"
    if csrc.is_dir():
        if (csrc / "RasterizeToPixels3DGSBwd.cu").is_file():
            return "modern"
    if (source / "csrc" / "rasterize_to_pixels_bwd.cu").is_file():
        return "legacy"
    raise SystemExit(
        f"Unrecognised gsplat layout: cannot find backward kernel source in {source}"
    )


# ---------------------------------------------------------------------------
# Modern layout (gsplat >= 1.5.0)
# ---------------------------------------------------------------------------

def patch_modern(target_csrc: Path, target_cuda: Path) -> None:
    """Patch gsplat >= 1.5.0 (CamelCase files)."""
    # 1. Copy header into csrc (for ext.cpp to include)
    shutil.copy2(HEADER, target_csrc / HEADER.name)

    # 2. Patch the backward kernel file
    bwd = target_csrc / "RasterizeToPixels3DGSBwd.cu"

    # 2a. Add includes after the last existing include
    replace_once(
        bwd,
        '#include "Utils.cuh"',
        '#include "Utils.cuh"\n#include "r6a_debug_atomic.cuh"\n#include <cstdint>',
    )

    # 2b. Inject counter + host functions inside namespace gsplat,
    #     right after "namespace gsplat {" and before "namespace cg = ..."
    replace_once(
        bwd,
        "namespace gsplat {\n\nnamespace cg = cooperative_groups;",
        "namespace gsplat {\n" + COUNTER_AND_FUNCS + "\nnamespace cg = cooperative_groups;",
    )

    # 2c. Add atomicAdd counter inside the warp-leader block.
    replace_once(
        bwd,
        "                gpuAtomicAdd(v_opacities + g, v_opacity_local);\n"
        "            }",
        "                gpuAtomicAdd(v_opacities + g, v_opacity_local);\n\n"
        "                ::atomicAdd(&r6a_debug_count, 1ULL);\n"
        "            }",
    )

    # 3. Patch ext.cpp: add R6-A debug bindings.
    #    Forward-declare the functions — don't include the header because
    #    ext.cpp is in cuda/ while the header is in cuda/csrc/ and the build
    #    system doesn't add csrc/ to the host compiler's include path.
    ext = target_cuda / "ext.cpp"
    text = ext.read_text()
    if "r6a_reset_debug_counter" not in text:
        # Forward-declare the R6-A functions (defined in RasterizeToPixels3DGSBwd.cu)
        replace_once(
            ext,
            '#include <torch/extension.h>',
            '#include <torch/extension.h>\n'
            '#include <cstdint>\n\n'
            'namespace gsplat::r6a {\n'
            'void reset_debug_counter();\n'
            'uint64_t get_debug_counter();\n'
            '}',
        )
        # Add m.def bindings after the last m.def
        lines = ext.read_text().split("\n")
        last_mdef_idx = -1
        for i, line in enumerate(lines):
            if "m.def(" in line:
                last_mdef_idx = i
        if last_mdef_idx < 0:
            raise RuntimeError("Cannot find m.def in ext.cpp")
        j = last_mdef_idx
        while j < len(lines) and ");" not in lines[j]:
            j += 1
        insert_line = j + 1
        lines.insert(insert_line, '')
        lines.insert(insert_line + 1, '    m.def("r6a_reset_debug_counter", &gsplat::r6a::reset_debug_counter);')
        lines.insert(insert_line + 2, '    m.def("r6a_get_debug_counter", &gsplat::r6a::get_debug_counter);')
        ext.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Legacy layout (gsplat < 1.5.0)
# ---------------------------------------------------------------------------

def patch_legacy(target_csrc: Path) -> None:
    shutil.copy2(HEADER, target_csrc / HEADER.name)
    bwd = target_csrc / "rasterize_to_pixels_bwd.cu"

    replace_once(
        bwd,
        '#include "types.cuh"',
        '#include "types.cuh"\n#include "r6a_debug_atomic.cuh"\n#include <cstdint>',
    )

    replace_once(
        bwd,
        "namespace gsplat {\n",
        "namespace gsplat {\n" + COUNTER_AND_FUNCS + "\n",
    )

    replace_once(
        bwd,
        "                gpuAtomicAdd(v_opacities + g, v_opacity_local);\n"
        "            }",
        "                gpuAtomicAdd(v_opacities + g, v_opacity_local);\n\n"
        "                ::atomicAdd(&r6a_debug_count, 1ULL);\n"
        "            }",
    )

    ext = target_csrc / "ext.cpp"
    replace_once(
        ext,
        '#include "bindings.h"',
        '#include "bindings.h"\n#include <cstdint>\n\n'
        'namespace gsplat::r6a {\n'
        'void reset_debug_counter();\n'
        'uint64_t get_debug_counter();\n'
        '}',
    )
    replace_once(
        ext,
        '    m.def("rasterize_to_pixels_bwd", &gsplat::rasterize_to_pixels_bwd_tensor);',
        '    m.def("rasterize_to_pixels_bwd", &gsplat::rasterize_to_pixels_bwd_tensor);\n'
        '    m.def("r6a_reset_debug_counter", &gsplat::r6a::reset_debug_counter);\n'
        '    m.def("r6a_get_debug_counter", &gsplat::r6a::get_debug_counter);',
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="gsplat package/source tree")
    parser.add_argument("--output", type=Path, required=True,
                        help="new isolated package path")
    args = parser.parse_args()

    layout = detect_layout(args.source)
    print(f"Detected gsplat layout: {layout}")

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    shutil.copytree(args.source, args.output)
    so = args.output / "csrc.so"
    if so.exists():
        so.unlink()

    if layout == "modern":
        target_csrc = args.output / "cuda" / "csrc"
        target_cuda = args.output / "cuda"
        patch_modern(target_csrc, target_cuda)
    else:
        target_csrc = args.output / "csrc"
        patch_legacy(target_csrc)

    print(f"R6-A patched source ready: {args.output}")


if __name__ == "__main__":
    main()
