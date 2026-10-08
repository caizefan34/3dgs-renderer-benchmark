"""Compare full-loss full-resolution gradients with pixel-sampled raster VJPs."""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402


PARAMETER_NAMES = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio_metrics(reference, candidate):
    ref = reference.flatten()
    got = candidate.flatten()
    ref_norm = float(torch.linalg.vector_norm(ref))
    got_norm = float(torch.linalg.vector_norm(got))
    dot = float(torch.dot(ref, got))
    difference = float(torch.linalg.vector_norm(ref - got))
    return {"cosine": dot / max(ref_norm * got_norm, 1e-30),
            "relative_l2": difference / max(ref_norm, 1e-30),
            "norm_ratio": got_norm / max(ref_norm, 1e-30)}


def stratified_mask(height, width, probability, generator):
    tile_class = torch.randint(0, 4, ((height + 15) // 16, (width + 15) // 16),
                               device="cuda", generator=generator)
    tile_class = tile_class.repeat_interleave(16, dim=0).repeat_interleave(16, dim=1)[:height, :width]
    y = torch.arange(height, device="cuda")[:, None]
    x = torch.arange(width, device="cuda")[None, :]
    parity = 2 * (y % 2) + x % 2
    if probability == .5:
        return (parity % 2 == tile_class % 2).float() / probability
    if probability == .75:
        return (parity != tile_class).float() / probability
    raise ValueError(probability)


def main():
    ply = REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    expected_hash = "64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc"
    if sha256(ply) != expected_hash:
        raise ValueError("Model PLY changed")
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output/room_big_capture4100"), images="images",
        resolution=-1, white_background=False, data_device="cuda", eval=True)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=4100, shuffle=False)
    cameras = scene.getTrainCameras()
    params = [getattr(model, name) for name in PARAMETER_NAMES]
    rows = []
    for camera_index in (0, 54):
        camera = cameras[camera_index]
        image = render_fastgs(camera, model, pipe, torch.zeros(3, device="cuda"), .5)["render"]
        gt = camera.original_image
        loss = .8 * l1_loss(image, gt) + .2 * (1 - fast_ssim(image.unsqueeze(0), gt.unsqueeze(0)))
        pixel_gradient = torch.autograd.grad(loss, image, retain_graph=True)[0]
        reference = torch.autograd.grad(image, params, grad_outputs=pixel_gradient, retain_graph=True)
        direct = torch.autograd.grad(loss, params, retain_graph=True)
        parity = {name: ratio_metrics(a, b) for name, a, b in zip(PARAMETER_NAMES, direct, reference)}
        rows_by_probability = {}
        ref_xyz_norm = torch.linalg.vector_norm(reference[0], dim=1)
        top_k = max(1, len(ref_xyz_norm) // 100)
        ref_top = torch.topk(ref_xyz_norm, top_k).indices
        ref_top_mask = torch.zeros(len(ref_xyz_norm), dtype=torch.bool, device="cuda")
        ref_top_mask[ref_top] = True
        for probability in (.75, .5):
            samples = []
            for seed in range(20260927, 20260935):
                generator = torch.Generator(device="cuda").manual_seed(seed)
                mask = stratified_mask(camera.image_height, camera.image_width, probability, generator)
                sampled = torch.autograd.grad(image, params,
                                              grad_outputs=pixel_gradient * mask.unsqueeze(0),
                                              retain_graph=True)
                metrics = {name: ratio_metrics(a, b) for name, a, b in zip(PARAMETER_NAMES, reference, sampled)}
                got_xyz_norm = torch.linalg.vector_norm(sampled[0], dim=1)
                got_top = torch.topk(got_xyz_norm, top_k).indices
                metrics["xyz_top_1pct_overlap"] = float(ref_top_mask[got_top].float().mean())
                samples.append(metrics)
            rows_by_probability[str(probability)] = samples
        row = {"camera_name": camera.image_name, "camera_index": camera_index,
               "resolution": [camera.image_width, camera.image_height], "loss": float(loss.detach()),
               "full_image_vjp_vs_direct_loss_gradient": parity,
               "sampled_vjp": rows_by_probability}
        rows.append(row)
        print("SPARSE_ADJOINT_GRADIENT " + json.dumps({
            "camera_name": camera.image_name,
            "full_xyz_parity": parity["_xyz"],
            "xyz_cosine_median": {key: float(np.median([s["_xyz"]["cosine"] for s in value]))
                                  for key, value in rows_by_probability.items()},
            "xyz_top_1pct_overlap_median": {key: float(np.median([s["xyz_top_1pct_overlap"] for s in value]))
                                            for key, value in rows_by_probability.items()},
        }, sort_keys=True), flush=True)
    result = {"source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
              "model_ply_sha256": expected_hash, "gaussian_count": len(model.get_xyz),
              "camera_count": len(cameras), "rows": rows,
              "scope": "Frozen room@4100 full-resolution original loss. Full loss image gradient is computed, then a tile-stratified Horvitz-Thompson mask samples pixel rows of the raster Jacobian. This Python screen does not skip CUDA backward work, train a model, measure speed, or evaluate novel views."}
    Path("/root/fastgs_sparse_adjoint_gradient_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
