#!/usr/bin/env python3
"""Non-timing executability smoke for the P2-1A-R2 F9→native entry."""

import json
import os

import torch


def main() -> None:
    import build

    module = build.build_and_load_experimental_gaussian_render_inference_scene()
    op = torch.ops.experimental.higs_native_hierarchy_from_projected
    schema = str(op.default._schema)
    n, width, height = 64, 256, 192
    device = "cuda"
    visible_ids = torch.arange(n, device=device, dtype=torch.int64)
    means = torch.zeros((n, 3), device=device, dtype=torch.float32)
    means[:, 0] = torch.linspace(-0.9, 0.9, n, device=device)
    means[:, 1] = torch.linspace(-0.7, 0.7, n, device=device)
    means[:, 2] = 4.0
    quats = torch.zeros((n, 4), device=device, dtype=torch.float32)
    quats[:, 0] = 1.0
    scales = torch.full((n, 3), 0.14, device=device, dtype=torch.float32)
    opacities = torch.full((n,), 0.5, device=device, dtype=torch.float32)
    coeffs = torch.zeros((n, 16, 3), device=device, dtype=torch.float32)
    viewmats = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    ks = torch.tensor(
        [[[128.0, 0.0, 128.0], [0.0, 128.0, 96.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    cam_positions = module.higs_camera_positions_from_viewmats(viewmats)
    f9 = module.higs_gatherless_projected_producer(
        visible_ids, means, quats, scales, opacities, coeffs, viewmats, ks,
        cam_positions, width, height, 0.3, 0.01, 1.0e4, 0.0,
    )
    radii, means2d, depths, conics, opacities_eval, colors_eval = f9
    background = torch.tensor([0.05, 0.10, 0.15], device=device, dtype=torch.float32)
    rgb, alpha, diagnostics = op(
        visible_ids, radii, means2d, depths, conics, opacities_eval, colors_eval,
        width, height, 16, background, True,
    )
    torch.cuda.synchronize()
    result = {
        "module": module.__file__,
        "op_exists": hasattr(module, "higs_native_hierarchy_from_projected"),
        "schema": schema,
        "f9": [
            {"shape": list(x.shape), "dtype": str(x.dtype), "contiguous": x.is_contiguous()}
            for x in f9
        ],
        "rgb": {"shape": list(rgb.shape), "dtype": str(rgb.dtype), "finite": bool(torch.isfinite(rgb).all()), "contiguous": rgb.is_contiguous()},
        "alpha": {"shape": list(alpha.shape), "dtype": str(alpha.dtype), "finite": bool(torch.isfinite(alpha).all()), "contiguous": alpha.is_contiguous()},
        "diagnostics": [
            {"shape": list(x.shape), "dtype": str(x.dtype), "contiguous": x.is_contiguous()}
            for x in diagnostics
        ],
    }
    assert result["op_exists"]
    assert result["rgb"]["shape"] == [height, width, 3]
    assert result["alpha"]["shape"] == [height, width, 1]
    assert result["rgb"]["dtype"] == "torch.float32" and result["alpha"]["dtype"] == "torch.float32"
    assert result["rgb"]["finite"] and result["alpha"]["finite"]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
