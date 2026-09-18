#!/usr/bin/env python3
"""Apply the opt-in C17-1 implementation to a disposable gsplat 1.5.3 tree."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "third_party_patches" / "c17_1" / "c17_1_tile_local.cu.inc"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected anchor missing: {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_c17_1_patch.py /path/to/gsplat")
    gsplat = Path(sys.argv[1]).resolve()
    payload = PAYLOAD
    if not payload.exists():  # copied to an isolated remote staging directory
        payload = Path(__file__).resolve().parent / "c17_1" / "c17_1_tile_local.cu.inc"
    csrc = gsplat / "gsplat" / "cuda" / "csrc"
    include = gsplat / "gsplat" / "cuda" / "include"
    cuda = gsplat / "gsplat" / "cuda"
    rendering = gsplat / "gsplat" / "rendering.py"
    required = [csrc / "IntersectTile.cu", csrc / "Intersect.cpp", include / "Ops.h", cuda / "ext.cpp", cuda / "_wrapper.py", rendering]
    absent = [str(p) for p in required if not p.exists()]
    if absent:
        raise RuntimeError("not a gsplat source tree: " + ", ".join(absent))

    tile = csrc / "IntersectTile.cu"
    text = tile.read_text()
    payload = payload.read_text()
    if "c17_tile_local_count_kernel" not in text:
        marker = "} // namespace gsplat"
        if marker not in text:
            raise RuntimeError("IntersectTile.cu namespace marker missing")
        tile.write_text(text.replace(marker, payload + "\n" + marker, 1))

    header = csrc / "Intersect.h"
    declaration = """
void launch_c17_tile_local_count_kernel(
    const at::Tensor means2d, const at::Tensor radii, const at::Tensor depths,
    const at::optional<at::Tensor> image_ids, const uint32_t I,
    const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
    at::Tensor tiles_per_gauss, at::Tensor tile_counts);
void launch_c17_tile_local_write_kernel(
    const at::Tensor means2d, const at::Tensor radii, const at::Tensor depths,
    const at::optional<at::Tensor> image_ids, const uint32_t I,
    const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
    const at::Tensor tile_offsets, at::Tensor tile_cursors, at::Tensor local_order_keys,
    at::Tensor flatten_ids);
void c17_tile_local_sort(
    const int64_t n_isects, const int32_t n_segments, const at::Tensor tile_offsets,
    at::Tensor local_order_keys, at::Tensor flatten_ids,
    at::Tensor local_order_keys_sorted, at::Tensor flatten_ids_sorted);
"""
    replace_once(header, "} // namespace gsplat", declaration + "\n} // namespace gsplat")

    ops = include / "Ops.h"
    c17_decl = """
// C17-1: returns (tiles_per_gauss, direct tile_offsets, depth-sorted flatten_ids).
std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile_c17(
    const at::Tensor means2d, const at::Tensor radii, const at::Tensor depths,
    const at::optional<at::Tensor> image_ids, const at::optional<at::Tensor> gaussian_ids,
    const uint32_t I, const uint32_t tile_size, const uint32_t tile_width,
    const uint32_t tile_height);
"""
    replace_once(ops, "at::Tensor intersect_offset(", c17_decl + "\nat::Tensor intersect_offset(")

    c17_impl = """
std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile_c17(
    const at::Tensor means2d, const at::Tensor radii, const at::Tensor depths,
    const at::optional<at::Tensor> image_ids, const at::optional<at::Tensor> gaussian_ids,
    const uint32_t I, const uint32_t tile_size, const uint32_t tile_width,
    const uint32_t tile_height) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d); CHECK_INPUT(radii); CHECK_INPUT(depths);
    const bool packed = means2d.dim() == 2;
    if (packed) {
        TORCH_CHECK(image_ids.has_value() && gaussian_ids.has_value(),
                    "C17 packed path requires image_ids and gaussian_ids.");
        CHECK_INPUT(image_ids.value()); CHECK_INPUT(gaussian_ids.value());
    }
    const auto opt = depths.options();
    const int64_t n_elements = means2d.numel() / 2;
    const int64_t n_segments = int64_t(I) * tile_width * tile_height;
    TORCH_CHECK(n_segments <= INT32_MAX, "C17 tile count exceeds int32 range.");
    at::Tensor tiles_per_gauss = at::empty_like(depths, opt.dtype(at::kInt));
    at::Tensor tile_counts = at::zeros({n_segments}, opt.dtype(at::kInt));
    if (n_elements) {
        launch_c17_tile_local_count_kernel(means2d, radii, depths,
            packed ? image_ids : c10::nullopt, I, tile_size, tile_width, tile_height,
            tiles_per_gauss, tile_counts);
    }
    // This is the same unavoidable exact-size boundary as baseline's cumsum.item().
    const int64_t n_isects = n_segments ? at::sum(tile_counts).item<int64_t>() : 0;
    TORCH_CHECK(n_isects <= INT32_MAX, "C17 intersection count exceeds rasterizer int32 range.");
    at::Tensor tile_offsets = at::empty({n_segments}, opt.dtype(at::kInt));
    if (n_segments) {
        CUB_WRAPPER(cub::DeviceScan::ExclusiveSum, tile_counts.data_ptr<int32_t>(),
                    tile_offsets.data_ptr<int32_t>(), n_segments, at::cuda::getCurrentCUDAStream());
    }
    at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
    if (!n_isects) return std::make_tuple(tiles_per_gauss,
        tile_offsets.view({I, tile_height, tile_width}), flatten_ids);
    at::Tensor tile_cursors = at::zeros({n_segments}, opt.dtype(at::kInt));
    at::Tensor order_keys = at::empty({n_isects}, opt.dtype(at::kLong));
    launch_c17_tile_local_write_kernel(means2d, radii, depths,
        packed ? image_ids : c10::nullopt, I, tile_size, tile_width, tile_height,
        tile_offsets, tile_cursors, order_keys, flatten_ids);
    at::Tensor order_keys_sorted = at::empty_like(order_keys);
    at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
    c17_tile_local_sort(n_isects, static_cast<int32_t>(n_segments), tile_offsets,
        order_keys, flatten_ids, order_keys_sorted, flatten_ids_sorted);
    return std::make_tuple(tiles_per_gauss,
        tile_offsets.view({I, tile_height, tile_width}), flatten_ids_sorted);
}

"""
    cpp = csrc / "Intersect.cpp"
    # Keep the scan in ATen here.  CUB is used only by the CUDA TU's segmented
    # sort; including a CUDA CUB release from a host-only TU couples the build
    # to the system CCCL version unnecessarily.
    replace_once(cpp, "at::Tensor intersect_offset(", c17_impl + "at::Tensor intersect_offset(")
    cpp_text = cpp.read_text().replace("#include <cub/cub.cuh>\n", "")
    cpp_text = cpp_text.replace(
        "at::Tensor tile_offsets = at::empty({n_segments}, opt.dtype(at::kInt));\n"
        "    if (n_segments) {\n"
        "        CUB_WRAPPER(cub::DeviceScan::ExclusiveSum, tile_counts.data_ptr<int32_t>(),\n"
        "                    tile_offsets.data_ptr<int32_t>(), n_segments, at::cuda::getCurrentCUDAStream());\n"
        "    }",
        "at::Tensor tile_offsets = at::cumsum(tile_counts, 0);\n"
        "    tile_offsets.sub_(tile_counts);")
    cpp.write_text(cpp_text)

    ext = cuda / "ext.cpp"
    replace_once(ext, 'm.def("intersect_tile", &gsplat::intersect_tile);',
                 'm.def("intersect_tile", &gsplat::intersect_tile);\n    m.def("intersect_tile_c17", &gsplat::intersect_tile_c17);')

    wrapper = cuda / "_wrapper.py"
    wrapper_impl = """

@torch.no_grad()
def isect_tiles_c17(
    means2d: Tensor, radii: Tensor, depths: Tensor, tile_size: int,
    tile_width: int, tile_height: int, packed: bool = False,
    n_images: Optional[int] = None, image_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    \"\"\"C17-1 exact direct tile-local arena construction.

    Returns tiles-per-Gaussian, direct tile offsets, and depth-ordered IDs.
    Unlike ``isect_tiles``, its second return is offsets, never legacy keys.
    \"\"\"
    if packed:
        assert image_ids is not None and gaussian_ids is not None and n_images is not None
        I = n_images
    else:
        I = math.prod(means2d.shape[:-2])
    return _make_lazy_cuda_func("intersect_tile_c17")(
        means2d.contiguous(), radii.contiguous(), depths.contiguous(),
        image_ids.contiguous() if image_ids is not None else None,
        gaussian_ids.contiguous() if gaussian_ids is not None else None,
        I, tile_size, tile_width, tile_height)
"""
    replace_once(wrapper, "\n\n@torch.no_grad()\ndef isect_offset_encode(", wrapper_impl + "\n\n@torch.no_grad()\ndef isect_offset_encode(")

    rendering_text = rendering.read_text()
    rendering_text = rendering_text.replace("import math\n", "import math\nimport os\n", 1)
    rendering_text = rendering_text.replace("    isect_offset_encode,\n    isect_tiles,", "    isect_offset_encode,\n    isect_tiles,\n    isect_tiles_c17,", 1)
    old = """    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        segmented=segmented,
        packed=packed,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    # print(\"rank\", world_rank, \"Before isect_offset_encode\")
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))
"""
    new = """    c17_mode = os.getenv(\"C17_MODE\", \"C17_BASELINE\")
    if c17_mode == \"C17_TILE_LOCAL\":
        if segmented:
            raise ValueError(\"C17_TILE_LOCAL is independent of segmented baseline mode.\")
        tiles_per_gauss, isect_offsets, flatten_ids = isect_tiles_c17(
            means2d, radii, depths, tile_size, tile_width, tile_height,
            packed=packed, n_images=I, image_ids=image_ids, gaussian_ids=gaussian_ids)
        isect_ids = None
    elif c17_mode == \"C17_BASELINE\":
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height,
            segmented=segmented, packed=packed, n_images=I,
            image_ids=image_ids, gaussian_ids=gaussian_ids)
        isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    else:
        raise ValueError(f\"unknown C17_MODE={c17_mode!r}; use C17_BASELINE or C17_TILE_LOCAL\")
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))
"""
    if old not in rendering_text and "c17_mode = os.getenv" not in rendering_text:
        raise RuntimeError("rendering.py standard 3DGS intersection anchor missing")
    if "c17_mode = os.getenv" not in rendering_text:
        rendering.write_text(rendering_text.replace(old, new, 1))
    print(f"C17-1 patched: {gsplat}")


if __name__ == "__main__":
    main()
