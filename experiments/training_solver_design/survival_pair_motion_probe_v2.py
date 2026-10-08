"""Fixed-state full-resolution movement audit for the survival-pair hypothesis.

The same 256 room@2000 parents move symmetrically on their longest 3D axis.
No parameter update occurs. A union tile list, adjacent independent child list,
and physical-depth independent child list isolate coverage, gating and ordering.
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


ROOT = Path(__file__).parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(image, reference):
    diff = (image - reference).detach()
    mag = diff.abs().amax(dim=-1)
    return {"max_abs": float(mag.max()), "mean_abs": float(diff.abs().mean()),
            "fraction_pixels_gt_1e-4": float((mag > 1e-4).float().mean()),
            "fraction_pixels_gt_1e-3": float((mag > 1e-3).float().mean())}


def split_direction(model, selected):
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci
    cov, _ = quat_scale_to_covar_preci(
        model.get_rotation[selected], model.get_scaling[selected],
        compute_covar=True, compute_preci=False, triu=False)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    return eigvecs[:, :, -1], eigvals[:, -1].clamp_min(0).sqrt()


def projected_children(model, selected, axis, long_scale, shift, cam, common):
    from gsplat.cuda._wrapper import fully_fused_projection, spherical_harmonics
    parent = model.get_xyz[selected]
    offset = shift * long_scale[:, None] * axis
    means = torch.stack((parent - offset, parent + offset), dim=1).reshape(-1, 3)
    quats = model.get_rotation[selected].repeat_interleave(2, dim=0)
    scales = model.get_scaling[selected].repeat_interleave(2, dim=0)
    q = model.get_opacity[selected].repeat_interleave(2)
    sh = model.get_features[selected].repeat_interleave(2, dim=0)
    view = cam.viewmatrix[None]
    radii, xy, depths, conics, _ = fully_fused_projection(
        means, None, quats, scales, view, cam.K[None],
        cam.image_width, cam.image_height, eps2d=common.EPS2D,
        packed=False, opacities=q)
    visible = (radii > 0).all(-1)
    colors = (spherical_harmonics(model.active_sh_degree, means, view, sh,
                                  masks=visible) + .5).relu()
    # A grouped primitive may still be listed on the parent's old tiles when
    # one child leaves the camera. Projected-invisible children must contribute
    # zero alpha, just as they would in an independent native tile list.
    active_q = torch.where(visible[0], q, 0.)
    rows = torch.cat((xy[0], conics[0], colors[0], active_q[:, None]), dim=-1).contiguous()
    return rows, radii, xy, depths, conics, active_q[None], int((~visible).sum())


def child_isects(projection, width, height):
    from gsplat.cuda._wrapper import isect_tiles
    rows, radii, xy, depths, conics, opacity, _ = projection
    tw, th = (width + 15) // 16, (height + 15) // 16
    _, keys, ids = isect_tiles(xy, radii, depths, 16, tw, th,
                               packed=False, n_images=1,
                               conics=conics, opacities=opacity)
    return keys.cpu().numpy(), ids.cpu().numpy()


def lists(old_keys, old_ids, child_keys, child_ids, selected, parent_depths,
          width, height, base_count):
    """Merge native intersection keys, preserving unedited original entries."""
    tw, th = (width + 15) // 16, (height + 15) // 16
    ntiles = tw * th
    old_tile = old_keys >> np.int64(32)
    child_tile = child_keys >> np.int64(32)
    select_mask = np.isin(old_ids, selected)
    assert np.all(parent_depths[child_ids // 2] >= 0), "visible group anchors must have nonnegative depth"
    parent_depth_bits = parent_depths.view(np.uint32).astype(np.int64)
    mapped_parent = selected[child_ids // 2]
    mapped_keys = (child_tile << np.int64(32)) | parent_depth_bits[child_ids // 2]
    # One group per (tile, parent), even if both offspring touch that tile.
    new_pair = np.stack((mapped_keys, mapped_parent), axis=1)
    new_pair = np.unique(new_pair, axis=0)
    old_selected = set(zip(old_tile[select_mask].tolist(), old_ids[select_mask].tolist()))
    need = np.array([(int(key >> np.int64(32)), int(idx)) not in old_selected
                     for key, idx in new_pair], dtype=np.bool_)
    added = new_pair[need]
    group_keys = np.concatenate((old_keys, added[:, 0]))
    group_ids = np.concatenate((old_ids, added[:, 1]))
    # Native key packs tile, then float-bit depth. Stable ordering keeps the
    # original tie order; newly inserted group entries follow exact ties.
    def ordered(keys, ids):
        perm = np.argsort(keys, kind="stable")
        keys, ids = keys[perm], ids[perm]
        offsets = np.searchsorted(keys, np.arange(ntiles, dtype=np.int64) << np.int64(32))
        assert len(offsets) == ntiles and np.all(np.diff(offsets) >= 0)
        return torch.from_numpy(offsets).cuda(), torch.from_numpy(ids.astype(np.int32)).cuda(), keys, ids
    group = ordered(group_keys, group_ids)
    nonparent_keys, nonparent_ids = old_keys[~select_mask], old_ids[~select_mask]
    child_global = base_count + child_ids
    physical = ordered(np.concatenate((nonparent_keys, child_keys)),
                       np.concatenate((nonparent_ids, child_global)))
    adjacent_keys = (child_tile << np.int64(32)) | parent_depth_bits[child_ids // 2]
    adjacent = ordered(np.concatenate((nonparent_keys, adjacent_keys)),
                       np.concatenate((nonparent_ids, child_global)))

    # Verify the group includes every tile in each child's native list.
    group_presence = set(zip((group[2] >> np.int64(32)).tolist(), group[3].tolist()))
    missing = sum((int(tile), int(parent)) not in group_presence
                  for tile, parent in zip(child_tile, mapped_parent))
    assert missing == 0, missing

    # A pair's offspring must be adjacent in each tile for a grouped draw to
    # match independent depth compositing, apart from the atomic gate policy.
    positions = {}
    for i, (key, idx) in enumerate(zip(physical[2], physical[3])):
        if idx >= base_count:
            tile, child = int(key >> np.int64(32)), int(idx - base_count)
            positions.setdefault((tile, child // 2), {})[child % 2] = i
    both = [pos for pos in positions.values() if len(pos) == 2]
    separated = [abs(pos[0] - pos[1]) - 1 for pos in both if abs(pos[0] - pos[1]) > 1]
    diagnostic = {"old_parent_tile_entries": int(select_mask.sum()),
                  "new_union_group_tile_entries": int(len(added)),
                  "offspring_tile_entries": int(len(child_ids)),
                  "both_children_in_tile": len(both),
                  "interleaved_by_other_gaussians": len(separated),
                  "interleaving_fraction": len(separated) / max(1, len(both)),
                  "max_intervening_gaussians": max(separated, default=0),
                  "tile_coverage_missing": missing}
    return group, adjacent, physical, diagnostic


def full_loss(common, image, target):
    return float(common.fused_total_loss(image.clamp(0, 1), target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    torch.set_num_threads(4)
    sys.path.insert(0, "/root/epic05_p11")
    import p11_common as common
    gpu = common.assert_clean_gpu(args.gpu)
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    uuid = gpu["gpu_info"].split(",")[0].strip()
    owners = {row.split(",")[0].strip() for row in common.gpu_compute_apps() if uuid in row}
    assert len(owners) == 1, owners
    identity = common.bootstrap_gmain()
    common.import_recipe()
    from gaussian_model import GaussianModel
    from sort_reuse_renderer import prepare, intersections
    from survival_pair_probe import native
    import survival_pair_renderer as pair

    state_path = Path(common.STATEGEN_DIR) / "room" / "state_2000.pt"
    selected = torch.tensor(json.loads((ROOT / "room_split_candidate_ids.json").read_text()),
                            dtype=torch.long, device="cuda")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model = GaussianModel(max_sh_degree=3)
    model.restore(state, common.TRAIN_CFG)
    del state
    axis, long_scale = split_direction(model, selected)
    sequence = np.load(state_path.with_name("camera_sequence.npy"))
    view_ids = list(dict.fromkeys(sequence[2000:].tolist()))[:args.views]
    dataset = common.load_dataset("room")
    result = {"completed": False, "scope": "fixed state; no training; deterministic ±k sigma-long-axis displacement",
              "gpu": gpu, "renderer_identity": identity, "checkpoint_sha256": sha(state_path),
              "sources": {name: sha(ROOT / name) for name in
                          ("survival_pair_motion_probe.py", "survival_pair_probe.py",
                           "survival_pair_renderer.py", "survival_pair_cuda.cu",
                           "sort_reuse_renderer.py", "room_split_candidate_ids.json")},
              "n_selected": len(selected), "view_ids": view_ids,
              "shifts_sigma": [0., .1, .3, .45], "views": []}
    started = time.perf_counter()
    with torch.no_grad():
        for view_id in view_ids:
            cam, target = dataset.get_item(view_id)
            data = prepare(model, cam)
            old = intersections(data, cam.image_width, cam.image_height)
            base = torch.cat((data["xy"][0], data["conics"][0], data["colors"][0],
                              data["opacity"].reshape(-1, 1)), dim=-1).contiguous()
            n = len(base)
            bg = torch.zeros(3, dtype=torch.float32, device="cuda")
            baseline, _ = native(base, cam.image_width, cam.image_height, old[3], old[2], bg)
            baseline_loss = full_loss(common, baseline, target)
            old_keys, old_ids = old[1].cpu().numpy(), old[2].cpu().numpy()
            depths = data["depths"][0][selected].float().cpu().numpy()
            # The same 3D anchor is used even when its original projected
            # ellipse is out of view and no native tile entry was generated.
            parent_camera = model.get_xyz[selected] @ cam.viewmatrix[:3, :3].T + cam.viewmatrix[:3, 3]
            invisible = ~(data["radii"][0][selected] > 0).all(-1)
            depths[invisible.cpu().numpy()] = parent_camera[invisible, 2].float().cpu().numpy()
            chosen = selected.cpu().numpy()
            entry = {"view_id": view_id, "width": cam.image_width, "height": cam.image_height,
                     "N": n, "baseline_loss": baseline_loss,
                     "old_intersection_count": len(old_ids),
                     "originally_invisible_selected": int(invisible.sum()), "shifts": []}
            result["views"].append(entry)
            for shift in result["shifts_sigma"]:
                proj = projected_children(model, selected, axis, long_scale, shift, cam, common)
                child_rows = proj[0]
                child_keys, child_ids = child_isects(proj, cam.image_width, cam.image_height)
                group_list, adjacent_list, physical_list, cover = lists(
                    old_keys, old_ids, child_keys, child_ids, chosen, depths,
                    cam.image_width, cam.image_height, n)
                children = child_rows.reshape(-1, 2, 9)
                group_mapping = torch.full((n,), -1, dtype=torch.int32, device="cuda")
                group_mapping[selected] = torch.arange(len(selected), dtype=torch.int32, device="cuda")
                group_image, _ = pair.rasterize(base, children, group_mapping,
                                                group_list[0], group_list[1],
                                                cam.image_width, cam.image_height, bg)
                full_rows = torch.cat((base, child_rows), dim=0).contiguous()
                phantom = child_rows[:, None].expand(-1, 2, -1).clone()
                phantom[:, 1, 8] = 0.
                child_mapping = torch.full((n + len(child_rows),), -1, dtype=torch.int32, device="cuda")
                child_mapping[n:] = torch.arange(len(child_rows), dtype=torch.int32, device="cuda")
                adjacent_image, _ = pair.rasterize(full_rows, phantom, child_mapping,
                                                   adjacent_list[0], adjacent_list[1],
                                                   cam.image_width, cam.image_height, bg)
                physical_image, _ = pair.rasterize(full_rows, phantom, child_mapping,
                                                   physical_list[0], physical_list[1],
                                                   cam.image_width, cam.image_height, bg)
                q = child_rows[:, 8]
                controls = {}
                for name, q_child in (("same_q", q), ("sixty_percent_q", .6 * q),
                                      ("center_corrected_q", q / (1 + torch.sqrt(1 - q)))):
                    control_rows = full_rows.clone()
                    control_rows[n:, 8] = q_child
                    img, _ = native(control_rows, cam.image_width, cam.image_height,
                                    physical_list[0], physical_list[1], bg)
                    controls[name] = {"versus_baseline": metrics(img, baseline),
                                      "loss_delta": full_loss(common, img, target) - baseline_loss}
                impact = ((group_image - baseline).abs().amax(-1) > 1e-5) | \
                         ((physical_image - baseline).abs().amax(-1) > 1e-5)
                region = {"pixel_fraction": float(impact.float().mean())}
                if impact.any():
                    for name, image in (("baseline", baseline), ("group", group_image),
                                        ("adjacent", adjacent_image), ("physical", physical_image)):
                        region[name + "_mse"] = float(((image.clamp(0, 1)[impact] - target[impact]) ** 2).mean())
                arm = {"shift_sigma": shift, "projected_invisible_children": proj[6],
                       "coverage_and_order": cover,
                       "group_vs_baseline": metrics(group_image, baseline),
                       "adjacent_vs_baseline": metrics(adjacent_image, baseline),
                       "physical_vs_baseline": metrics(physical_image, baseline),
                       "group_vs_adjacent": metrics(group_image, adjacent_image),
                       "adjacent_vs_physical": metrics(adjacent_image, physical_image),
                       "group_vs_physical": metrics(group_image, physical_image),
                       "group_loss_delta": full_loss(common, group_image, target) - baseline_loss,
                       "adjacent_loss_delta": full_loss(common, adjacent_image, target) - baseline_loss,
                       "physical_loss_delta": full_loss(common, physical_image, target) - baseline_loss,
                       "impact_region": region, "matched_geometry_standard_controls": controls}
                entry["shifts"].append(arm)
                Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
                print(f"view={view_id} shift={shift} union_add={cover['new_union_group_tile_entries']} "
                      f"interleaved={cover['interleaving_fraction']:.3f} "
                      f"group-vs-physical-max={arm['group_vs_physical']['max_abs']:.5g} "
                      f"roi={region['pixel_fraction']:.4f}", flush=True)
            Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            current = {row.split(",")[0].strip() for row in common.gpu_compute_apps() if uuid in row}
            assert current == owners, (current, owners)
    result["completed"] = True
    result["wall_seconds"] = time.perf_counter() - started
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
