#!/usr/bin/env python3
"""Isolated regression self-test for D0 tracer paths the 3000-iter smoke
cannot reach: per-uid array growth (_ensure), death registration with tensor
reason codes, opacity-reset counters, and finalize() artifact writing.

No renderer, tiny tensors, seconds to run.  On EPIC-05:
  CUDA_VISIBLE_DEVICES=2 /root/miniforge3/envs/gsplat/bin/python d0_selftest.py
Exit 0 iff every check passes.
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d0_trace import D0Tracer, N_FEATURES, TRAJ_FIELDS  # noqa: E402

tmp = tempfile.mkdtemp(prefix="d0_selftest_")
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name +
          ("  " + str(detail) if detail != "" else ""), flush=True)
    if not ok:
        fails.append(name)
    return ok


tr = D0Tracer(None, "selftest", tmp, n_cameras=311, grad_threshold=8e-4,
              sample_offset=11, cost_samples=False)
cap0 = tr.cap
tr.register_init(10)
check("init registered", tr.next_uid == 10 and tr.cap == cap0, (tr.next_uid, cap0))
check("init parent fill", int(tr.u_parent[5].item()) == -1)

# --- growth: two doublings, data preserved --------------------------------
n = 700000
ids = torch.arange(10, n, dtype=torch.int64, device=tr.dev)
parents = torch.full_like(ids, 7)
feat = torch.full((N_FEATURES, ids.numel()), 3.25, dtype=torch.float32,
                  device=tr.dev)
tr.register_births(ids, parents, 1, 1234, feat)
check("grow cap >= n", tr.cap >= n, tr.cap)
check("grow next_uid", tr.next_uid == n, tr.next_uid)
check("grow keeps init rows", bool((tr.u_parent[:10] == -1).all()))
check("grow birth_iter", int(tr.u_birth_iter[500000].item()) == 1234)
check("grow feat", abs(float(tr.u_feat[17, 400000].item()) - 3.25) < 1e-6)
check("grow birth_type", int(tr.u_birth_type[400000].item()) == 1)
check("grow birth_event", int(tr.u_birth_event[400000].item()) == 0)
check("grow untouched fill", int(tr.u_death_iter[300000].item()) == -1 and
      float(tr.tiles_cum[300000].item()) == 0.0)

# --- deaths: tensor reasons + counters ------------------------------------
tr.cur = {k: 0 for k in TRAJ_FIELDS}
died = torch.tensor([100, 200, 300, 400], dtype=torch.int64, device=tr.dev)
reasons = torch.tensor([1, 2, 3, 4], dtype=torch.uint8, device=tr.dev)
tr.register_deaths(died, reasons, 777)
c = tr.cur
check("death counters",
      c["prunes_opacity"] == 1 and c["prunes_screen"] == 1 and
      c["prunes_world"] == 1 and c["split_parent_removed"] == 1 and
      c["prunes_total"] == 3, dict(c))
check("death iter", int(tr.u_death_iter[200].item()) == 777)
check("death reasons", [int(x) for x in tr.u_reason[died].cpu()] == [1, 2, 3, 4])

tr.cur = {k: 0 for k in TRAJ_FIELDS}
tr.register_deaths(torch.tensor([500], device=tr.dev), 1, 800)
tr.register_deaths(torch.tensor([600], device=tr.dev), 4, 800)
check("int death counters", tr.cur["prunes_total"] == 1 and
      tr.cur["prunes_opacity"] == 1 and tr.cur["split_parent_removed"] == 1,
      dict(tr.cur))

# --- opacity reset (with and without an open iteration row) ---------------
tr.register_opacity_reset(3000)
check("reset list", tr.reset_iters == [3000])
tr.register_opacity_reset(3000)
tr.cur["opacity_reset"] = 0
tr.register_opacity_reset(3000)
check("reset counter in cur", tr.cur["opacity_reset"] == 1)
tr.cur = None

# --- finalize --------------------------------------------------------------
tr.begin_event(1)
tr.traj["iter"].extend([1, 2])
tr.traj["N_t"].extend([10, 20])
tr.traj["loss"].extend([0.5, 0.25])
tr.checkpoints.append({"iter": 5000, "psnr": 20.0})
for it in (11, 31):
    tr.sample_summary.append({"iter": it, "cam_idx": 0, "I_t": 5,
                              "sum_tiles": 5, "N_t": 10, "n_visible": 3,
                              "n_nonzero": 2, "tiles_uid_sum": 5.0})
meta = tr.finalize({"arm": "selftest"})
check("finalize n_uids", meta["n_uids"] == n, meta["n_uids"])
check("finalize births", meta["n_births_post_init"] == n - 10,
      meta["n_births_post_init"])
check("finalize sumcheck", meta["n_sumcheck_mismatches"] == 0)
for fn in ("d0_uid_table.npz", "d0_birth_records.npz",
           "d0_population_trajectory.npz", "d0_sample_summary.npz",
           "d0_trace_meta.json"):
    check("finalize file " + fn, os.path.exists(os.path.join(tmp, fn)))
ut = np.load(os.path.join(tmp, "d0_uid_table.npz"))
check("uid_table roundtrip", ut["uid"].shape[0] == n and
      int(ut["death_iter"][200]) == 777)
br = np.load(os.path.join(tmp, "d0_birth_records.npz"), allow_pickle=True)
check("birth_records roundtrip", br["uid"].shape[0] == n - 10 and
      br["feat"].shape == (N_FEATURES, n - 10))

# --- model-side densify/prune path (M3/M4/M5/M6) ---------------------------
import torch.nn as nn          # noqa: E402
import torch.optim as optim    # noqa: E402
import d0_gaussian_model as d0gm  # noqa: E402

N = 64


def make_model():
    m = d0gm.GaussianModel(max_sh_degree=0)
    m._xyz = nn.Parameter(torch.randn(N, 3, device="cuda") * 0.1)
    m._shs = nn.Parameter(torch.randn(N, 1, 3, device="cuda"))
    m._opacity = nn.Parameter(torch.full((N,), 2.0, device="cuda"))
    m._scaling = nn.Parameter(torch.full((N, 3), -1.0, device="cuda"))
    m._rotation = nn.Parameter(torch.zeros(N, 4, device="cuda"))
    with torch.no_grad():
        m._rotation[:, 0] = 1.0
    m.xyz_gradient_accum = torch.zeros(N, 1, device="cuda")
    m.denom = torch.ones(N, 1, device="cuda")
    m.max_radii2D = torch.zeros(N, device="cuda")
    m.tmp_radii = None
    m.optimizer = optim.Adam([
        {"params": [m._xyz], "lr": 0.01, "name": "xyz"},
        {"params": [m._shs], "lr": 0.01, "name": "shs"},
        {"params": [m._opacity], "lr": 0.01, "name": "opacity"},
        {"params": [m._scaling], "lr": 0.01, "name": "scaling"},
        {"params": [m._rotation], "lr": 0.01, "name": "rotation"},
    ])
    m._gaussian_ids = torch.arange(N, dtype=torch.long, device="cuda")
    m._next_gaussian_id = N
    m.d0_iter = 3100
    return m


def attach(m, tag):
    t = D0Tracer(m, tag, tmp, n_cameras=8, grad_threshold=8e-4,
                 sample_offset=11, cost_samples=False)
    t.register_init(N)
    m.d0 = t
    t.cur = {k: 0 for k in TRAJ_FIELDS}
    return t


# A: densify + prune with max_screen_size=None (iterations <= 3000 path)
mp = make_model()
tr2 = attach(mp, "selftest_modelA")
with torch.no_grad():
    mp.xyz_gradient_accum[0:16, 0] = 0.01   # split (0:8) + clone (8:16)
    mp._scaling[8:16] = -5.0                # small -> clone-eligible
    mp._opacity[30:34] = -10.0              # opacity prune
ev = mp.densify_and_prune(max_grad=8e-4, min_opacity=0.005, extent=10.0,
                          max_screen_size=None,
                          radii=torch.zeros(N, device="cuda"))
check("A ev counts", ev["cloned"] == 8 and ev["split"] == 8 and
      ev["pruned_total"] == 4 and ev["pruned_opacity"] == 4,
      {k: ev[k] for k in ("cloned", "split", "pruned_total", "pruned_opacity")})
check("A death reasons",
      [int(tr2.u_reason[i].item()) for i in range(8)] == [4] * 8 and
      [int(tr2.u_reason[i].item()) for i in range(30, 34)] == [1] * 4)
check("A death iter", int(tr2.u_death_iter[30].item()) == 3100)
check("A counters",
      tr2.cur["prunes_opacity"] == 4 and tr2.cur["prunes_screen"] == 0 and
      tr2.cur["prunes_world"] == 0 and tr2.cur["split_parent_removed"] == 8 and
      tr2.cur["births_clone"] == 8 and tr2.cur["births_split"] == 16 and
      tr2.cur["prunes_total"] == 4, dict(tr2.cur))
check("A uids", int(mp._gaussian_ids.shape[0]) == 64 + 8 + 16 - 8 - 4)
mpost = tr2.u_birth_type[:tr2.next_uid] > 0
check("A birth records", int(mpost.sum().item()) == 24, int(mpost.sum().item()))
check("A clone parents",
      bool((tr2.u_parent[64:72] == torch.arange(8, 16, device="cuda")).all()) and
      bool((tr2.u_parent[72:88] ==
            torch.cat([torch.arange(0, 8, device="cuda")] * 2)).all()))
check("A birth iter", int(tr2.u_birth_iter[64].item()) == 3100)

# B: prune path with max_screen_size set (iterations > 3000, M6 tensor codes).
# Note: reason 2 (screen) cannot fire through this path by construction — the
# unconditional clone densification_postfix zeroes max_radii2D before the
# prune mask is built (frozen reference behavior; confirmed against the P12
# reference runs: all non-opacity prunes are world prunes from iter 3100 on).
# Reason 2 is covered by the pure-tracer tensor-reason check above.
mp2 = make_model()
tr3 = attach(mp2, "selftest_modelB")
with torch.no_grad():
    mp2.max_radii2D[20:24] = 1000.0   # would be screen prune, but reset below
    mp2._scaling[24:28] = 5.0         # world prune  (reason 3)
    mp2._opacity[30:34] = -10.0       # opacity prune(reason 1)
ev2 = mp2.densify_and_prune(max_grad=8e-4, min_opacity=0.005, extent=10.0,
                            max_screen_size=100.0,
                            radii=torch.zeros(N, device="cuda"))
check("B ev counts", ev2["cloned"] == 0 and ev2["split"] == 0 and
      ev2["pruned_total"] == 8 and ev2["pruned_opacity"] == 4 and
      ev2["pruned_screen"] == 0 and ev2["pruned_world"] == 4, ev2)
check("B death reasons",
      [int(tr3.u_reason[i].item()) for i in range(24, 28)] == [3] * 4 and
      [int(tr3.u_reason[i].item()) for i in range(30, 34)] == [1] * 4 and
      bool((tr3.u_death_iter[20:24] == -1).all()))
check("B counters",
      tr3.cur["prunes_opacity"] == 4 and tr3.cur["prunes_screen"] == 0 and
      tr3.cur["prunes_world"] == 4 and tr3.cur["split_parent_removed"] == 0 and
      tr3.cur["prunes_total"] == 8, dict(tr3.cur))
check("B uids", int(mp2._gaussian_ids.shape[0]) == N - 8)

print("SELFTEST", "PASS" if not fails else "FAIL",
      "fails=%d" % len(fails), flush=True)
sys.exit(0 if not fails else 1)
