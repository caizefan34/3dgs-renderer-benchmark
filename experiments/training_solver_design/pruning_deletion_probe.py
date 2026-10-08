"""Fixed-state full-resolution pruning opportunity check on an existing P11 state.

This is an expensive per-Gaussian removal oracle, not a deployable score. The B
views are withheld from scoring here but may have appeared in checkpoint training.
No optimizer step, density event, or official unseen-view quality claim is made.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def measure(model, views, common):
    losses, images = [], []
    with torch.no_grad():
        for cam, gt in views:
            _, image, _, _ = common.render_with_meta(
                model, cam, model.active_sh_degree, absgrad=False
            )
            losses.append(float(common.fused_total_loss(image, gt)))
            images.append(image.detach())
    return losses, images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room")
    ap.add_argument("--iteration", type=int, default=10000)
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--candidates", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(4)
    sys.path.insert(0, "/root/epic05_p11")
    import p11_common as C

    gpu = C.assert_clean_gpu(args.gpu)
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    uuid = gpu["gpu_info"].split(",")[0].strip()
    owners = [row for row in C.gpu_compute_apps() if uuid in row]
    assert len(owners) == 1, owners
    driver_pid = owners[0].split(",")[0].strip()

    def assert_exclusive():
        rows = [row for row in C.gpu_compute_apps() if uuid in row]
        assert {row.split(",")[0].strip() for row in rows} == {driver_pid}, rows

    identity = C.bootstrap_gmain()
    C.import_recipe()
    from gaussian_model import GaussianModel
    started = time.perf_counter()
    state_path = Path(C.STATEGEN_DIR) / args.scene / f"state_{args.iteration}.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert state["iteration"] == args.iteration
    seq_path = state_path.with_name("camera_sequence.npy")
    sequence = np.load(seq_path).tolist()
    unique = list(dict.fromkeys(sequence[args.iteration:]))
    a_ids, b_ids = unique[:3], unique[3:6]
    assert len(b_ids) == 3 and not set(a_ids) & set(b_ids)
    dataset = C.load_dataset(args.scene)
    a_views = [dataset.get_item(i) for i in a_ids]
    b_views = [dataset.get_item(i) for i in b_ids]
    model = GaussianModel(max_sh_degree=3)
    model.restore(state, C.TRAIN_CFG)
    del state
    opacity = model.get_opacity.detach().flatten()
    base_a, base_a_images = measure(model, a_views, C)
    base_b, base_b_images = measure(model, b_views, C)
    replay_a, replay_a_images = measure(model, a_views, C)
    replay_b, replay_b_images = measure(model, b_views, C)
    replay = {
        "loss_delta_A": [x - y for x, y in zip(replay_a, base_a)],
        "loss_delta_B": [x - y for x, y in zip(replay_b, base_b)],
        "image_max_abs_A": [float((x - y).abs().max()) for x, y in zip(replay_a_images, base_a_images)],
        "image_max_abs_B": [float((x - y).abs().max()) for x, y in zip(replay_b_images, base_b_images)],
    }
    # First-order predicted deletion loss from the ordinary opacity-logit VJP:
    # dL/d(logit q) = q(1-q) dL/dq, hence delta q=-q predicts
    # delta L = -dL/d(logit q)/(1-q). No extra loss or renderer approximation.
    influence = torch.zeros_like(opacity)
    for cam, gt in a_views:
        _, image, _, _ = C.render_with_meta(
            model, cam, model.active_sh_degree, absgrad=False
        )
        loss = C.fused_total_loss(image, gt)
        logit_grad = torch.autograd.grad(loss, model._opacity)[0].detach().flatten()
        influence += -logit_grad / (1.0 - opacity).clamp_min(1e-8)
    influence /= len(a_views)

    radii_sum = torch.zeros_like(opacity)
    seen_a = torch.zeros_like(opacity, dtype=torch.int32)
    with torch.no_grad():
        for cam, _ in a_views:
            _, _, radii, _ = C.render_with_meta(
                model, cam, model.active_sh_degree, absgrad=False
            )
            visible = (radii[0] > 0).any(-1)
            seen_a += visible.int()
            radii_sum += radii[0].float().clamp_min(0).square().sum(-1)
    pool = ((opacity >= 0.005) & (opacity < 0.02) & (seen_a >= 2)).nonzero().flatten()
    assert len(pool) >= args.candidates * 4, (len(pool), args.candidates)
    rng = np.random.default_rng(270927)
    pool_np = pool.cpu().numpy()
    random_ids = rng.choice(pool_np, size=args.candidates // 2, replace=False)
    cheap = (opacity * radii_sum).detach()
    top_ids = pool[torch.topk(cheap[pool], args.candidates // 4).indices].cpu().numpy()
    low_ids = pool[torch.topk(cheap[pool], args.candidates // 4, largest=False).indices].cpu().numpy()
    ids = list(dict.fromkeys(np.concatenate([random_ids, top_ids, low_ids]).tolist()))
    assert len(ids) >= args.batch_size * 3
    assert_exclusive()

    records = []
    original = model._opacity.data
    for index, idx in enumerate(ids):
        old = original[idx].clone()
        original[idx] = -30.0
        try:
            changed_a, images_a = measure(model, a_views, C)
            changed_b, images_b = measure(model, b_views, C)
        finally:
            original[idx] = old
        delta_a = [x - y for x, y in zip(changed_a, base_a)]
        delta_b = [x - y for x, y in zip(changed_b, base_b)]
        image_a = [float((x - y).abs().mean()) for x, y in zip(images_a, base_a_images)]
        image_b = [float((x - y).abs().mean()) for x, y in zip(images_b, base_b_images)]
        records.append({
            "id": idx,
            "opacity": float(opacity[idx]),
            "cheap_opacity_area": float(cheap[idx]),
            "first_order_deletion_A": float(influence[idx]),
            "seen_A": int(seen_a[idx]),
            "loss_delta_A": delta_a,
            "loss_delta_B": delta_b,
            "image_l1_delta_A": image_a,
            "image_l1_delta_B": image_b,
        })
        if (index + 1) % 20 == 0:
            print(f"{args.scene}: {index + 1}/{len(ids)}", flush=True)
            assert_exclusive()

    def batch_score(chosen):
        selected = torch.as_tensor(chosen, device="cuda", dtype=torch.long)
        old = original[selected].clone()
        original[selected] = -30.0
        try:
            a, _ = measure(model, a_views, C)
            b, _ = measure(model, b_views, C)
        finally:
            original[selected] = old
        return {
            "ids": chosen,
            "loss_delta_A": [x - y for x, y in zip(a, base_a)],
            "loss_delta_B": [x - y for x, y in zip(b, base_b)],
        }

    k = args.batch_size
    by_a = sorted(records, key=lambda r: np.mean(r["loss_delta_A"]))[:k]
    by_opacity = sorted(records, key=lambda r: r["opacity"])[:k]
    by_area = sorted(records, key=lambda r: r["cheap_opacity_area"])[:k]
    by_influence = sorted(records, key=lambda r: r["first_order_deletion_A"])[:k]
    batches = {
        "deletion_loss_A_oracle": batch_score([r["id"] for r in by_a]),
        "opacity": batch_score([r["id"] for r in by_opacity]),
        "opacity_area": batch_score([r["id"] for r in by_area]),
        "first_order_deletion_A": batch_score([r["id"] for r in by_influence]),
    }
    assert_exclusive()
    torch.cuda.synchronize()
    result = {
        "scope": "Full-resolution fixed-state removal oracle; no training speed/quality claim",
        "scene": args.scene,
        "iteration": args.iteration,
        "gpu": {**gpu, "driver_pid": driver_pid},
        "renderer_identity": identity,
        "environment": C.env_identity(),
        "source_sha256": sha(__file__),
        "state_sha256": sha(state_path),
        "sequence_sha256": sha(seq_path),
        "A": a_ids,
        "B": b_ids,
        "protocol": {
            "loss": "0.8 L1 + 0.2 fused D-SSIM, full 1920x1080",
            "candidate_pool": "opacity in [0.005,0.02), visible in at least two A views",
            "B_status": "withheld from diagnostic score, may occur in checkpoint prehistory",
            "single_deletion": "set one raw opacity logit to -30; no retraining",
            "first_order_score": "-dL/dlogit(opacity)/(1-opacity), averaged over three A views",
            "batch_deletion": f"{k} candidates selected from the same pool",
            "not_production_score": True,
        },
        "base_loss_A": base_a,
        "base_loss_B": base_b,
        "no_edit_replay": replay,
        "pool_count": len(pool),
        "candidates": records,
        "batches": batches,
        "wall_seconds": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"WROTE {out} ({len(records)} candidates)", flush=True)


if __name__ == "__main__":
    main()
