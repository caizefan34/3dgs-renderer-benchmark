#!/usr/bin/env python3
"""Create an isolated, R6-B-patched gsplat 1.5.3 source tree.

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="gsplat package/source tree containing cuda/csrc")
    parser.add_argument("--output", type=Path, required=True,
                        help="new isolated package path, conventionally .../gsplat")
    args = parser.parse_args()
    csrc = args.source / "cuda" / "csrc"
    if not (csrc / "rasterize_to_pixels_bwd.cu").is_file():
        raise SystemExit("--source must be a gsplat 1.5.x package/source tree")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    shutil.copytree(args.source, args.output)
    target = args.output / "cuda" / "csrc"
    shutil.copy2(HEADER, target / HEADER.name)
    replace_once(
        target / "rasterize_to_pixels_bwd.cu",
        '#include "types.cuh"',
        '#include "types.cuh"\n#include "r6b_persistent_buffers.cuh"',
    )
    replace_once(
        target / "rasterize_to_pixels_bwd.cu",
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
        target / "rasterize_to_pixels_bwd.cu",
        "    return std::make_tuple(\n        v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities\n    );",
        "    if (r6b_enabled) r6b::raster_buffers().finish(flatten_ids);\n"
        "    return std::make_tuple(\n        v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities\n    );",
    )
    replace_once(
        target / "ext.cpp",
        '    m.def("rasterize_to_pixels_bwd", &gsplat::rasterize_to_pixels_bwd_tensor);',
        '    m.def("rasterize_to_pixels_bwd", &gsplat::rasterize_to_pixels_bwd_tensor);\n'
        '    m.def("r6b_set_mode", &gsplat::r6b::set_mode);\n'
        '    m.def("r6b_mode", &gsplat::r6b::mode);\n'
        '    m.def("r6b_invalidate", &gsplat::r6b::invalidate);\n'
        '    m.def("r6b_last_prepare_ms", &gsplat::r6b::last_prepare_ms);',
    )
    replace_once(
        target / "ext.cpp",
        '#include "bindings.h"',
        '#include "bindings.h"\n\nnamespace gsplat::r6b {\n'
        'void set_mode(int mode);\nint mode();\nvoid invalidate();\nfloat last_prepare_ms();\n}',
    )
    print(f"R6-B patched source ready: {args.output}")


if __name__ == "__main__":
    main()
