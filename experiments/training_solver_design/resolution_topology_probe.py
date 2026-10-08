"""Compare full/half-resolution densification evidence on a fixed P11 state.

This is a cross-resolution candidate diagnostic, not a paper-protocol training
result. It leaves the checkpoint, optimizer, topology, and trainer untouched.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlap(reference, candidate):
    intersection = int((reference & candidate).sum())
    a = int(reference.sum())
    b = int(candidate.sum())
    return {"reference": a, "candidate": b, "intersection": intersection,
            "recall": intersection / a if a else None,
            "precision": intersection / b if b else None,
            "false_positives": b - intersection}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("room", "garden", "bicycle"), required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(4)
    sys.path.insert(0, "/root/epic05_p11")
    import p11_common as C

    gpu = C.assert_clean_gpu(args.gpu)
    uuid = gpu["gpu_info"].split(",")[0].strip()
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    rows = [row for row in C.gpu_compute_apps() if uuid in row]
    assert len(rows) == 1, rows
    driver_pid = rows[0].split(",")[0].strip()
    gpu.update(driver_pid=driver_pid, namespace_pid=os.getpid())

    def exclusive():
        rows = [row for row in C.gpu_compute_apps() if uuid in row]
        assert {row.split(",")[0].strip() for row in rows} == {driver_pid}, rows

    identity = C.bootstrap_gmain()
    C.import_recipe()
    from gaussian_model import GaussianModel
    checkpoint = Path(C.STATEGEN_DIR) / args.scene / "state_2000.pt"
    sequence_path = checkpoint.with_name("camera_sequence.npy")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["iteration"] == 2000
    model = GaussianModel(max_sh_degree=3)
    model.restore(saved, C.TRAIN_CFG)
    dataset = C.load_dataset(args.scene)
    sequence = np.load(sequence_path)[2000:]
    camera_ids = list(dict.fromkeys(map(int, sequence)))[:args.views]
    assert len(camera_ids) == args.views
    n = len(model.get_xyz)
    scores = {scale: torch.zeros(n, device="cuda") for scale in (1, 2)}
    seen = {scale: torch.zeros(n, device="cuda") for scale in (1, 2)}
    times = {scale: [] for scale in (1, 2)}
    per_view = []

    for camera_id in camera_ids:
        exclusive()
        cam, gt = dataset.get_item(camera_id)
        one = {"camera_id": camera_id}
        for factor in (1, 2):
            model.optimizer.zero_grad(set_to_none=True)
            if factor == 1:
                view, target = cam, gt
            else:
                K = cam.K.clone()
                K[:2] /= factor
                view = SimpleNamespace(K=K, viewmatrix=cam.viewmatrix,
                                       image_width=cam.image_width // factor,
                                       image_height=cam.image_height // factor)
                target = F.interpolate(gt.permute(2, 0, 1)[None],
                                       size=(view.image_height, view.image_width),
                                       mode="bilinear", align_corners=False)[0].permute(1, 2, 0).contiguous()
            torch.cuda.synchronize()
            start = time.perf_counter()
            _, image, radii, xy = C.render_with_meta(model, view, model.active_sh_degree, absgrad=False)
            loss = C.fused_total_loss(image, target)
            loss.backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            grad = xy.grad[0].detach().clone()
            grad[:, 0] *= view.image_width / 2.0
            grad[:, 1] *= view.image_height / 2.0
            visible = (radii[0] > 0).all(-1)
            scores[factor] += grad.norm(dim=-1) * visible
            seen[factor] += visible
            times[factor].append(elapsed)
            one[str(factor)] = {"loss": float(loss), "visible": int(visible.sum()),
                                "render_backward_seconds": elapsed}
            del image, radii, xy, loss, grad
        per_view.append(one)
        print("VIEW", args.scene, camera_id, flush=True)

    averages = {factor: scores[factor] / seen[factor].clamp_min(1) for factor in (1, 2)}
    valid = (seen[1] > 0) | (seen[2] > 0)
    full = averages[1]
    half = averages[2]
    threshold = 0.0002  # pinned gsplat DefaultStrategy.grow_grad2d
    full_candidate = valid & (full > threshold)
    half_candidate = valid & (half > threshold)
    threshold_comparison = overlap(full_candidate, half_candidate)
    k = int(full_candidate.sum())
    budget = {}
    if k:
        eligible = torch.where(valid)[0]
        target_ids = eligible[torch.topk(full[eligible], k).indices]
        half_ids = eligible[torch.topk(half[eligible], k).indices]
        target_mask = torch.zeros(n, device="cuda", dtype=torch.bool)
        half_mask = torch.zeros_like(target_mask)
        target_mask[target_ids] = True
        half_mask[half_ids] = True
        budget = overlap(target_mask, half_mask)
    exclusive()
    result = {
        "scope": "Fixed P11 state resolution/topology proxy; not exact paper trainer, no training or quality claim",
        "scene": args.scene, "state_step": 2000, "camera_ids": camera_ids,
        "checkpoint_sha256": sha(checkpoint), "camera_sequence_sha256": sha(sequence_path),
        "source_sha256": sha(__file__), "common_sha256": sha(C.__file__),
        "renderer_identity": identity, "gpu": gpu, "num_gaussians": n,
        "protocol": {"full_resolution": [cam.image_width, cam.image_height],
                     "half_gt": "bilinear align_corners=False as paper trainer",
                     "loss": "P11 original fused L1+D-SSIM",
                     "gradient": "signed projected means2d gradient norm, normalized to [-1,1] screen coordinates",
                     "threshold": threshold, "fixed_weights": True,
                     "visibility": "both radii axes positive",
                     "selection": "per-view norm averaged over visible sampled views"},
        "per_view": per_view,
        "threshold_candidate": threshold_comparison,
        "same_point_budget": budget,
        "observed_seconds": {str(factor): sum(times[factor]) for factor in (1, 2)},
        "completed": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("COMPLETE", args.scene, args.out, flush=True)


if __name__ == "__main__":
    main()
