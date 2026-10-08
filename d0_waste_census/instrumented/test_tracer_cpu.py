#!/usr/bin/env python3
"""CPU dry-run of the D0 tracer: lifecycle, shapes, NaN policy, artifacts.

No training semantics here -- this only shakes out tracer runtime bugs using a
stub model + synthetic captures, mirroring the frozen call order:
  register_init -> (per iter) note_render -> [densify: capture_birth_features,
  register_births, register_deaths] -> note_iteration -> finalize
"""
import json
import os
import tempfile

import numpy as np
import torch

from d0_trace import D0Tracer, N_FEATURES, FEATURE_NAMES

H, W, TS = 64, 96, 16
TH, TW = H // TS, W // TS


class StubModel:
    def __init__(self, n):
        self._xyz = torch.randn(n, 3)
        self._scaling = torch.randn(n, 3) * 0.1 - 2.0
        self._opacity = torch.randn(n)
        self._gaussian_ids = torch.arange(n, dtype=torch.long)
        self._next_gaussian_id = n
        self.xyz_gradient_accum = torch.rand(n, 1)
        self.denom = torch.ones(n, 1)
        self.max_radii2D = torch.rand(n) * 50
        self.active_sh_degree = 1

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)


def grow_model(model, n_new):
    for attr in ("_xyz", "_scaling", "_opacity"):
        t = getattr(model, attr)
        setattr(model, attr, torch.cat([t, torch.randn(n_new, *t.shape[1:])]))
    model.xyz_gradient_accum = torch.cat(
        [model.xyz_gradient_accum, torch.rand(n_new, 1)])
    model.denom = torch.cat([model.denom, torch.ones(n_new, 1)])
    model.max_radii2D = torch.cat([model.max_radii2D, torch.rand(n_new) * 50])


class StubProxy:
    def __init__(self, n):
        self.absgrad = torch.ones(1, n, 2) * 0.01
        self.grad = torch.ones(1, n, 2) * 0.001


def make_capture(model, vis_ids, n_isects=None):
    vis_ids = torch.as_tensor(vis_ids, dtype=torch.long)
    V = vis_ids.numel()
    tiles = torch.zeros(V)
    if V:
        tiles = torch.randint(0, 4, (V,)).float()
        tiles[0] = max(tiles[0], 1)
    n = int(tiles.sum().item())
    N = model._xyz.shape[0]
    rfull = torch.zeros(N, 2)
    rfull[vis_ids] = 20.0
    m2d = torch.zeros(V, 2)
    if V:
        m2d[:, 0] = torch.rand(V) * W
        m2d[:, 1] = torch.rand(V) * H
    return {
        "visible_ids": vis_ids,
        "isect_row_space": "visible",
        "tiles_per_gauss": tiles,
        "depths_vis": torch.rand(V) * 5 + 1,
        "means2d_px_visible": m2d,
        "radii_full": rfull,
        "means2d_proxy": StubProxy(N),
        "n_isects": n if n_isects is None else n_isects,
        "n_isects_full": n,
        "f9_enabled": True,
        "width": float(W), "height": float(H),
        "cam_center": torch.zeros(3),
    }


def main():
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp(prefix="d0_cpu_")
    model = StubModel(10)
    d0 = D0Tracer(model, "room", tmp, n_cameras=17,
                  grad_threshold=0.0008, sample_offset=11, device="cpu")
    d0.register_init(10)

    img = torch.rand(H, W, 3)
    gt = torch.rand(H, W, 3)
    phase = {k: [1.0] for k in ("forward", "loss", "backward", "densify",
                                "optimizer")}

    # iter 1: no densify, one render
    d0.note_render(1, 0, make_capture(model, range(10)), img, gt)
    d0.note_iteration(1, 0.5, phase, 10.0)
    assert d0.f9_enabled is True and d0.m2d_mode == "pixel"

    # iter 2: clone 3 parents (rows 1,3,5)
    d0.begin_event(2)
    d0.note_render(2, 2, make_capture(model, range(10)), img, gt)
    rows = torch.tensor([1, 3, 5])
    feat = d0.capture_birth_features(rows, 1, 2)
    assert feat.shape == (N_FEATURES, 3)
    assert not torch.isnan(feat[0]).any(), "parent_age must be defined"
    assert not torch.isnan(feat[1]).any(), "parent_opacity must be defined"
    assert not torch.isnan(feat[17]).any(), "vis_count_32 must be defined"
    new_ids = torch.arange(10, 13)
    model._gaussian_ids = torch.cat([model._gaussian_ids, new_ids])
    grow_model(model, 3)
    d0.register_births(new_ids, torch.tensor([1, 3, 5]), 1, 2, feat)
    d0.note_iteration(2, 0.4, phase, 10.0)
    assert d0.u_birth_type[10].item() == 1
    assert d0.u_parent[11].item() == 3
    assert d0.cur is None

    # iter 3: split parents rows 0 and 2 -> 4 children, parents die reason 4
    d0.begin_event(3)
    d0.note_render(3, 3, make_capture(model, [0, 2, 5, 7]), img, gt)
    rows = torch.tensor([0, 2])
    feat = d0.capture_birth_features(rows, 2, 3).repeat(1, 2)
    assert feat.shape == (N_FEATURES, 4)
    assert feat[0, 3].item() == feat[0, 1].item(), "child block ordering"
    child_ids = torch.arange(13, 17)
    model._gaussian_ids = torch.cat([model._gaussian_ids, child_ids])
    grow_model(model, 4)
    d0.register_births(child_ids, torch.tensor([0, 2, 0, 2]), 2, 3, feat)
    d0.register_deaths(torch.tensor([0, 2]), 4, 3)
    d0.note_iteration(3, 0.3, phase, 10.0)
    assert d0.u_death_iter[0].item() == 3 and d0.u_reason[0].item() == 4
    assert d0.u_parent[14].item() == 2 and d0.u_parent[15].item() == 0
    assert d0.u_parent[16].item() == 2
    assert d0.traj["split_parent_removed"][-1] == 2

    # iter 4: opacity prune (uid 5) + screen prune (uid 7), n_isects mismatch
    d0.begin_event(4)
    d0.note_render(4, 4, make_capture(model, [5, 7], n_isects=99), img, gt)
    assert len(d0.sumcheck_mismatch) == 1
    d0.register_deaths(torch.tensor([5, 7]), torch.tensor([1, 2], dtype=torch.uint8), 4)
    d0.note_iteration(4, 0.2, phase, 10.0)
    assert d0.u_reason[5].item() == 1 and d0.u_reason[7].item() == 2
    assert d0.cur is None

    # iter 5: opacity reset + sampled cost dump at iteration 11 offset
    d0.register_opacity_reset(5)
    assert d0.reset_iters == [5]

    for t in (11, 31):
        d0.note_render(t, 1, make_capture(model, [10, 11, 13]), img, gt)
        d0.note_iteration(t, 0.1, phase, 9.0)
    assert len(d0.sample_summary) == 2, d0.sample_summary
    assert os.path.exists(os.path.join(tmp, "d0_cost_samples", "sample_00011.npz"))

    meta = d0.finalize({"scene": "room"})
    assert meta["n_features"] == N_FEATURES == 30
    assert meta["n_births_post_init"] == 3 + 4
    for f in ("d0_uid_table.npz", "d0_birth_records.npz",
              "d0_population_trajectory.npz", "d0_sample_summary.npz",
              "d0_trace_meta.json"):
        assert os.path.exists(os.path.join(tmp, f)), f

    uid = np.load(os.path.join(tmp, "d0_uid_table.npz"))
    assert uid["uid"].shape[0] == 17
    assert uid["tiles_cum"][10] > 0
    br = np.load(os.path.join(tmp, "d0_birth_records.npz"), allow_pickle=True)
    assert br["feat"].shape == (N_FEATURES, 7)
    assert list(br["feature_names"]) == FEATURE_NAMES
    traj = np.load(os.path.join(tmp, "d0_population_trajectory.npz"))
    assert traj["iter"].tolist() == [1, 2, 3, 4, 11, 31]
    assert traj["I_t"].sum() == sum(t for t in (
        d0.traj["I_t"]))

    # wrong row-space must abort
    bad = make_capture(model, [1, 2])
    bad["isect_row_space"] = "full"
    d2 = D0Tracer(model, "room", tmp, n_cameras=17, grad_threshold=0.0008,
                  sample_offset=11, device="cpu")
    d2.register_init(10)
    try:
        d2.note_render(1, 0, bad, img, gt)
        raise AssertionError("space guard did not fire")
    except AssertionError as e:
        assert "row space" in str(e)

    meta2 = json.load(open(os.path.join(tmp, "d0_trace_meta.json")))
    assert meta2["means2d_mode"] == "pixel"
    print("CPU dry-run OK")
    print(f"  F={N_FEATURES}, uids=17, sampled cost dumps=2, tmp={tmp}")


if __name__ == "__main__":
    main()
