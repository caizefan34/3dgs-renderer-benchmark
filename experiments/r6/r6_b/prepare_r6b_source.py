#!/usr/bin/env python3
"""Create an isolated, R6-B-patched gsplat source tree.

Supports two gsplat source layouts:

  Legacy (< 1.5.0):
    csrc/rasterize_to_pixels_bwd.cu  with torch::zeros_like and #include "types.cuh"
    csrc/ext.cpp                     with m.def("rasterize_to_pixels_bwd", ...)

  Modern (>= 1.5.0, e.g. 1.5.3 on A100):
    csrc/Rasterization.cpp           with at::zeros_like and #include "Rasterization.h"
    cuda/ext.cpp                     with m.def("rasterize_to_pixels_3dgs_bwd", ...)

The baseline site package is never modified.  Point --source at a gsplat
source/package tree; set PYTHONPATH to the parent of --output when running the
benchmark or trainer so its JIT backend builds the isolated tree.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HEADER = ROOT / "r6b_persistent_buffers.cuh"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected R6-B patch context missing: {path}")
    path.write_text(text.replace(old, new, 1))


def detect_layout(csrc: Path, cuda_parent: Path) -> str:
    """Return 'legacy' or 'modern' based on which files exist."""
    if (csrc / "rasterize_to_pixels_bwd.cu").is_file():
        return "legacy"
    if (csrc / "Rasterization.cpp").is_file():
        return "modern"
    raise SystemExit(
        f"Unrecognised gsplat layout: neither rasterize_to_pixels_bwd.cu nor "
        f"Rasterization.cpp found in {csrc}"
    )


def patch_modern(target_csrc: Path, cuda_dir: Path) -> None:
    """Patch gsplat >= 1.5.0 (CamelCase files, at::zeros_like)."""
    # 1. Copy header into csrc
    shutil.copy2(HEADER, target_csrc / HEADER.name)

    # 2. Patch Rasterization.cpp: add include + replace allocation + add finish()
    rast = target_csrc / "Rasterization.cpp"
    replace_once(
        rast,
        '#include "Rasterization.h"',
        '#include "Rasterization.h"\n#include "r6b_persistent_buffers.cuh"',
    )
    replace_once(
        rast,
        "    at::Tensor v_means2d = at::zeros_like(means2d);\n"
        "    at::Tensor v_conics = at::zeros_like(conics);\n"
        "    at::Tensor v_colors = at::zeros_like(colors);\n"
        "    at::Tensor v_opacities = at::zeros_like(opacities);\n"
        "    at::Tensor v_means2d_abs;\n"
        "    if (absgrad) {\n"
        "        v_means2d_abs = at::zeros_like(means2d);\n"
        "    }",
        "    const bool r6b_enabled = r6b::raster_buffers().enabled_for(means2d);\n"
        "    std::array<at::Tensor, 5> r6b_outputs;\n"
        "    if (r6b_enabled) {\n"
        "        r6b_outputs = r6b::raster_buffers().prepare(\n"
        "            means2d, conics, colors, opacities, absgrad\n"
        "        );\n"
        "    }\n"
        "    at::Tensor v_means2d = r6b_enabled ? r6b_outputs[0] : at::zeros_like(means2d);\n"
        "    at::Tensor v_conics = r6b_enabled ? r6b_outputs[1] : at::zeros_like(conics);\n"
        "    at::Tensor v_colors = r6b_enabled ? r6b_outputs[2] : at::zeros_like(colors);\n"
        "    at::Tensor v_opacities = r6b_enabled ? r6b_outputs[3] : at::zeros_like(opacities);\n"
        "    at::Tensor v_means2d_abs = absgrad\n"
        "        ? (r6b_enabled ? r6b_outputs[4] : at::zeros_like(means2d))\n"
        "        : at::Tensor();",
    )
    # 3. Add finish() call before the return in rasterize_to_pixels_3dgs_bwd.
    #    The return is:
    #        return std::make_tuple(
    #            v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities
    #        );
    #    This pattern appears in both the 3dgs_bwd and 2dgs_bwd and from_world
    #    functions, so we must match the 3dgs one specifically.
    replace_once(
        rast,
        "    return std::make_tuple(\n"
        "        v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities\n"
        "    );\n"
        "}\n"
        "\n"
        "std::tuple<at::Tensor, at::Tensor> rasterize_to_indices_3dgs(",
        "    if (r6b_enabled) r6b::raster_buffers().finish(\n"
        "        flatten_ids, means2d.size(0) * means2d.size(1));\n"
        "    return std::make_tuple(\n"
        "        v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities\n"
        "    );\n"
        "}\n"
        "\n"
        "std::tuple<at::Tensor, at::Tensor> rasterize_to_indices_3dgs(",
    )

    # 4. Patch cuda/ext.cpp: add R6-B bindings
    ext = cuda_dir / "ext.cpp"
    replace_once(
        ext,
        '    m.def(\n'
        '        "rasterize_to_pixels_3dgs_bwd", &gsplat::rasterize_to_pixels_3dgs_bwd\n'
        '    );',
        '    m.def(\n'
        '        "rasterize_to_pixels_3dgs_bwd", &gsplat::rasterize_to_pixels_3dgs_bwd\n'
        '    );\n'
        '    m.def("r6b_set_mode", &gsplat::r6b::set_mode);\n'
        '    m.def("r6b_mode", &gsplat::r6b::mode);\n'
        '    m.def("r6b_invalidate", &gsplat::r6b::invalidate);\n'
        '    m.def("r6b_last_prepare_ms", &gsplat::r6b::last_prepare_ms);\n'
        '    m.def("r6b_last_scatter_ms", &gsplat::r6b::last_scatter_ms);\n'
        '    m.def("r6b_last_clear_ms", &gsplat::r6b::last_clear_ms);\n'
        '    m.def("r6b_metadata_bytes", &gsplat::r6b::metadata_bytes);\n'
        '    m.def("r6b_prev_n_rows", &gsplat::r6b::prev_n_rows);',
    )
    replace_once(
        ext,
        '#include <torch/extension.h>',
        '#include <torch/extension.h>\n\n'
        'namespace gsplat::r6b {\n'
        'void set_mode(int mode);\nint mode();\nvoid invalidate();\n'
        'float last_prepare_ms();\nfloat last_scatter_ms();\n'
        'float last_clear_ms();\nint64_t metadata_bytes();\nint64_t prev_n_rows();\n}',
    )


def patch_legacy(target_csrc: Path, cuda_parent: Path) -> None:
    """Patch gsplat < 1.5.0 (snake_case files, torch::zeros_like)."""
    shutil.copy2(HEADER, target_csrc / HEADER.name)
    bwd = target_csrc / "rasterize_to_pixels_bwd.cu"
    replace_once(
        bwd,
        '#include "types.cuh"',
        '#include "types.cuh"\n#include "r6b_persistent_buffers.cuh"',
    )
    replace_once(
        bwd,
        "    torch::Tensor v_means2d = torch::zeros_like(means2d);\n"
        "    torch::Tensor v_conics = torch::zeros_like(conics);\n"
        "    torch::Tensor v_colors = torch::zeros_like(colors);\n"
        "    torch::Tensor v_opacities = torch::zeros_like(opacities);\n"
        "    torch::Tensor v_means2d_abs;\n"
        "    if (absgrad) {\n"
        "        v_means2d_abs = torch::zeros_like(means2d);\n"
        "    }",
        "    const bool r6b_enabled = r6b::raster_buffers().enabled_for(means2d);\n"
        "    std::array<torch::Tensor, 5> r6b_outputs;\n"
        "    if (r6b_enabled) {\n"
        "        r6b_outputs = r6b::raster_buffers().prepare(\n"
        "            means2d, conics, colors, opacities, absgrad\n"
        "        );\n"
        "    }\n"
        "    torch::Tensor v_means2d = r6b_enabled ? r6b_outputs[0] : torch::zeros_like(means2d);\n"
        "    torch::Tensor v_conics = r6b_enabled ? r6b_outputs[1] : torch::zeros_like(conics);\n"
        "    torch::Tensor v_colors = r6b_enabled ? r6b_outputs[2] : torch::zeros_like(colors);\n"
        "    torch::Tensor v_opacities = r6b_enabled ? r6b_outputs[3] : torch::zeros_like(opacities);\n"
        "    torch::Tensor v_means2d_abs = absgrad\n"
        "        ? (r6b_enabled ? r6b_outputs[4] : torch::zeros_like(means2d))\n"
        "        : torch::Tensor();",
    )
    replace_once(
        bwd,
        "    return std::make_tuple(\n        v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities\n    );",
        "    if (r6b_enabled) r6b::raster_buffers().finish(\n"
        "        flatten_ids, means2d.size(0) * means2d.size(1));\n"
        "    return std::make_tuple(\n        v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities\n    );",
    )
    ext = target_csrc / "ext.cpp"
    replace_once(
        ext,
        '    m.def("rasterize_to_pixels_bwd", &gsplat::rasterize_to_pixels_bwd_tensor);',
        '    m.def("rasterize_to_pixels_bwd", &gsplat::rasterize_to_pixels_bwd_tensor);\n'
        '    m.def("r6b_set_mode", &gsplat::r6b::set_mode);\n'
        '    m.def("r6b_mode", &gsplat::r6b::mode);\n'
        '    m.def("r6b_invalidate", &gsplat::r6b::invalidate);\n'
        '    m.def("r6b_last_prepare_ms", &gsplat::r6b::last_prepare_ms);\n'
        '    m.def("r6b_last_scatter_ms", &gsplat::r6b::last_scatter_ms);\n'
        '    m.def("r6b_last_clear_ms", &gsplat::r6b::last_clear_ms);\n'
        '    m.def("r6b_metadata_bytes", &gsplat::r6b::metadata_bytes);\n'
        '    m.def("r6b_prev_n_rows", &gsplat::r6b::prev_n_rows);',
    )
    replace_once(
        ext,
        '#include "bindings.h"',
        '#include "bindings.h"\n\nnamespace gsplat::r6b {\n'
        'void set_mode(int mode);\nint mode();\nvoid invalidate();\nfloat last_prepare_ms();\n'
        'float last_scatter_ms();\nfloat last_clear_ms();\nint64_t metadata_bytes();\nint64_t prev_n_rows();\n}',
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="gsplat package/source tree containing cuda/csrc")
    parser.add_argument("--output", type=Path, required=True,
                        help="new isolated package path, conventionally .../gsplat")
    args = parser.parse_args()

    csrc = args.source / "cuda" / "csrc"
    cuda_dir = args.source / "cuda"
    if not csrc.is_dir():
        raise SystemExit("--source must be a gsplat package/source tree with cuda/csrc")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    shutil.copytree(args.source, args.output)
    target_csrc = args.output / "cuda" / "csrc"
    target_cuda = args.output / "cuda"
    layout = detect_layout(csrc, cuda_dir)
    print(f"Detected gsplat layout: {layout}")

    if layout == "modern":
        patch_modern(target_csrc, target_cuda)
    else:
        patch_legacy(target_csrc, target_cuda)

    print(f"R6-B patched source ready: {args.output}")


if __name__ == "__main__":
    main()
