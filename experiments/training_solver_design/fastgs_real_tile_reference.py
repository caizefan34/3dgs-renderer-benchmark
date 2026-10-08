"""Read native FastGS CUDA buffers and recompose four real-camera pixels.

Diagnostic only: no author source or renderer behavior is modified.
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from diff_gaussian_rasterization_fastgs import _C  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def take(buffer: torch.Tensor, cursor: int, count: int, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    itemsize = torch.empty((), dtype=dtype).element_size()
    start = ((buffer.data_ptr() + cursor + 127) // 128) * 128 - buffer.data_ptr()
    end = start + count * itemsize
    if start < 0 or end > buffer.numel():
        raise AssertionError((start, end, buffer.numel()))
    return buffer.narrow(0, start, count * itemsize).view(dtype), end


def decode_geometry(buffer: torch.Tensor, points: int) -> dict:
    cursor = 0
    depths, cursor = take(buffer, cursor, points, torch.float32)
    clamped, cursor = take(buffer, cursor, points * 3, torch.uint8)
    radii, cursor = take(buffer, cursor, points, torch.int32)
    mean2d, cursor = take(buffer, cursor, points * 2, torch.float32)
    cov3d, cursor = take(buffer, cursor, points * 6, torch.float32)
    conic, cursor = take(buffer, cursor, points * 4, torch.float32)
    rgb, cursor = take(buffer, cursor, points * 3, torch.float32)
    tiles, cursor = take(buffer, cursor, points, torch.int32)
    return {
        "depths": depths,
        "radii": radii,
        "mean2d": mean2d.reshape(points, 2),
        "conic": conic.reshape(points, 4),
        "rgb": rgb.reshape(points, 3),
        "tiles": tiles,
    }


def decode_image(buffer: torch.Tensor, pixels: int) -> dict:
    cursor = 0
    alpha, cursor = take(buffer, cursor, pixels, torch.float32)
    n_contrib, cursor = take(buffer, cursor, pixels, torch.int32)
    ranges, cursor = take(buffer, cursor, pixels * 2, torch.int32)
    return {"alpha": alpha, "n_contrib": n_contrib, "ranges": ranges.reshape(pixels, 2)}


def recompose(x: int, y: int, means: torch.Tensor, conics: torch.Tensor,
              colors: torch.Tensor, bg: torch.Tensor) -> tuple[list[float], int, int, bool]:
    trans = 1.0
    rgb = [0.0, 0.0, 0.0]
    last = 0
    accepted = 0
    early_stop = False
    for index in range(len(means)):
        dx = float(means[index, 0]) - x
        dy = float(means[index, 1]) - y
        co = conics[index]
        power = -0.5 * (float(co[0]) * dx * dx + float(co[2]) * dy * dy) - float(co[1]) * dx * dy
        if power > 0:
            continue
        alpha = min(0.99, float(co[3]) * math.exp(power))
        if alpha < 1 / 255:
            continue
        next_trans = trans * (1 - alpha)
        if next_trans < 1e-4:
            early_stop = True
            break
        for channel in range(3):
            rgb[channel] += trans * alpha * float(colors[index, channel])
        trans = next_trans
        accepted += 1
        last = index + 1
    return [rgb[i] + trans * float(bg[i]) for i in range(3)], last, accepted, early_stop


def main() -> None:
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output/room_big_capture4100"),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=True,
    )
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=4100, shuffle=False)
    camera = scene.getTrainCameras()[0]
    captured = []
    original = _C.rasterize_gaussians

    def capture(*args):
        result = original(*args)
        captured.append(result)
        return result

    _C.rasterize_gaussians = capture
    try:
        image = render_fastgs(camera, model, pipe, torch.zeros(3, device="cuda"), 0.5)["render"]
        torch.cuda.synchronize()
    finally:
        _C.rasterize_gaussians = original
    if len(captured) != 1:
        raise AssertionError(len(captured))

    rendered, buckets, _, radii, geom_buffer, bin_buffer, img_buffer, _, _ = captured[0]
    n = int(model.get_xyz.shape[0])
    width, height = camera.image_width, camera.image_height
    geometry = decode_geometry(geom_buffer, n)
    image_state = decode_image(img_buffer, width * height)
    point_list, _ = take(bin_buffer, 0, rendered, torch.int32)
    tile_x, tile_y = width // 2 // 16, height // 2 // 16
    tile_id = tile_y * ((width + 15) // 16) + tile_x
    begin, end = image_state["ranges"][tile_id].cpu().tolist()
    ids = point_list[begin:end].long()
    means = geometry["mean2d"][ids].cpu()
    conics = geometry["conic"][ids].cpu()
    colors = geometry["rgb"][ids].cpu()
    bg = torch.zeros(3)
    pixels = []
    for y in (height // 2, height // 2 + 1):
        for x in (width // 2, width // 2 + 1):
            reference, last, accepted, stopped = recompose(x, y, means, conics, colors, bg)
            native = image[:, y, x].detach().cpu().tolist()
            native_last = int(image_state["n_contrib"][y * width + x].item())
            pixels.append({
                "xy": [x, y], "max_rgb_error": max(abs(a - b) for a, b in zip(reference, native)),
                "last_contributor": [last, native_last],
                "accepted_splats": accepted, "early_stop": stopped,
            })

    result = {
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "model_ply_sha256": sha256(REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"),
        "camera": camera.image_name,
        "resolution": [width, height], "gaussians": n,
        "rendered_tile_instances": rendered,
        "decoded_tiles_sum": int(geometry["tiles"].sum().item()),
        "positive_radii": int((radii > 0).sum().item()),
        "tile_id": tile_id, "tile_splats": end - begin,
        "pixels": pixels,
        "max_rgb_error": max(row["max_rgb_error"] for row in pixels),
        "last_contributor_matches": all(row["last_contributor"][0] == row["last_contributor"][1] for row in pixels),
    }
    if result["decoded_tiles_sum"] != rendered or not result["last_contributor_matches"] or result["max_rgb_error"] > 1e-4:
        raise AssertionError(result)
    print("FASTGS_REAL_TILE_REFERENCE " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
