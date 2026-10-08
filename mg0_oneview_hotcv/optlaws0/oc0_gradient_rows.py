#!/usr/bin/env python3
"""OPTLAWS0 / OC0 step 2 — cross-view gradient-conflict labels.

Usage: oc0_gradient_rows.py <room|bicycle> <20000|25000>

Reuses VERBATIM the MG0 machinery through mg0_core (frozen 64-camera set, N1
snapshot checkpoint, c0 render branch, P12 fused loss, Adam preconditioner
P_t).  Only the OC0 frozen 32-camera subset (OC0_SELECTION.json) and the OC0
frozen 2048 rows are used.

For every selected Gaussian j and camera v with a nonzero row-gradient norm,

    X[v,j,:] = P(g_v,j) ,  P_t(x) = x / (sqrt(v_t) + eps)

and for every camera pair (a,b) with BOTH row-gradient norms nonzero:

    cosine_ab = <X_a,X_b> / (||X_a|| ||X_b||)
    Y_ab      = 1 - cosine_ab               (primary continuous target)
    conflict_ab = cosine_ab < 0             (primary binary label)
    severe_ab   = cosine_ab < -0.25         (diagnostic; NOT tuned)

Per Gaussian also: negative-pair fraction, severe-pair fraction and directional
dispersion 1 - ||mean(normalized gradients)|| (mean over cameras with nonzero
row-gradient norm; NaN when no camera has a nonzero norm).

Emits
-----
  OC0_GRADIENT_ROWS_{scene}_{ckpt}.npz
      X_preconditioned_fp32 [nc,2048,59], row norms, per-pair cosines,
      pair index tables, per-Gaussian conflict aggregates, per-camera losses.

Telemetry only: no training, no densification, no test views, no threshold
tuning, no MG0 gate touched.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "MG0_SCRIPTS_DIR",
    "/root/epic05_mg0/scripts" if os.path.isdir("/root/epic05_mg0/scripts")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")))
import mg0_core as C  # noqa: E402
import oc0_common as K  # noqa: E402


def load_selection():
    with open(K.artifact("OC0_SELECTION.json")) as f:
        return json.load(f)


def main():
    scene, stage = sys.argv[1], int(sys.argv[2])
    assert scene in C.SCENES and stage in (20000, 25000)
    t0 = time.time()
    sel = load_selection()
    cell = sel["cells"][K.cell_tag(scene, stage)]
    cams = [int(v) for v in cell["oc0_camera_ids"]]
    sel_ids = [int(v) for v in cell["selection_ids"]]
    hot_ids = set(int(v) for v in cell["hot_ids"])
    nc, ng = len(cams), len(sel_ids)

    C.bootstrap_c0()
    fused_loss = C.make_fused_loss()
    dataset = C.load_scene_dataset(scene)
    ck = C.load_checkpoint(scene, stage)
    model = C.build_model(ck)
    C.scene_reset()
    del ck
    torch.cuda.empty_cache()
    denom = C.flat_denom(model)
    idx = torch.tensor(sel_ids, dtype=torch.int64, device="cuda")
    den_sel = denom.index_select(0, idx)
    print(f"[oc0_gradient_rows] {scene}@{stage} N={model._xyz.shape[0]} "
          f"nc={nc} ng={ng} mem={C.gpu_mem_gb():.1f}GB", flush=True)

    # float32, not float16: the frozen preconditioner P_t(x)=x/(sqrt(v_t)+eps)
    # sends a coordinate with a still-zero Adam second moment (row born at the
    # last densification, optimizer state padded with zeros) to g/eps, which
    # reaches |P| ~ 2e8 here -- beyond float16's 65504.  Storing those in fp16
    # produced inf and then NaN cosines for the affected rows.
    X = torch.empty((nc, ng, C.ROW_DIM), dtype=torch.float32)
    norms = torch.empty((nc, ng), dtype=torch.float32)
    losses = []
    for i, v in enumerate(cams):
        cam, gt = dataset.get_item(v)
        loss, g = C.view_grad_and_loss(model, cam, gt, fused_loss)
        losses.append(float(loss))
        P = g.index_select(0, idx) / den_sel
        X[i] = P.to("cpu", torch.float32)
        norms[i] = P.norm(dim=1).to("cpu", torch.float32)
        del g, P
        print(f"[oc0_gradient_rows] view {i + 1}/{nc} cam={v} loss={losses[-1]:.5f}",
              flush=True)
    assert bool(torch.isfinite(X).all()), "preconditioned rows not finite"

    # ---- pairwise cosines (pairs enumerated a<b in frozen camera order) -----
    Xg = X.to("cuda", torch.float32)                       # [nc,ng,59]
    nrm = norms.to("cuda")
    Xn = Xg / nrm.clamp_min(1e-30).unsqueeze(-1)           # zero rows -> zero
    nz = nrm > 0
    pairs = [(a, b) for a in range(nc) for b in range(a + 1, nc)]
    n_pair = len(pairs)
    pair_a = torch.tensor([p[0] for p in pairs], dtype=torch.int64)
    pair_b = torch.tensor([p[1] for p in pairs], dtype=torch.int64)
    cos = torch.zeros((n_pair, ng), dtype=torch.float32, device="cuda")
    for k, (a, b) in enumerate(pairs):
        cos[k] = (Xn[a] * Xn[b]).sum(-1)
    valid = (nz[pair_a].T & nz[pair_b].T)                  # [ng,n_pair]
    cos_t = cos.T                                          # [ng,n_pair]

    neg_frac = torch.zeros(ng)
    sev_frac = torch.zeros(ng)
    n_valid_pair = valid.sum(1)
    conflict = (cos_t < K.CONFLICT_COSINE) & valid
    severe = (cos_t < K.SEVERE_COSINE) & valid
    neg_frac = (conflict.sum(1).float() / n_valid_pair.clamp_min(1).float()).cpu()
    sev_frac = (severe.sum(1).float() / n_valid_pair.clamp_min(1).float()).cpu()

    # directional dispersion 1 - ||mean(normalized gradients)|| over nonzero cams
    # nz is [nc, ng]: the mean is taken over CAMERAS (dim 0) for each Gaussian,
    # so the count must also be per-Gaussian -- nz.sum(0), not nz.sum(1).
    nz_cnt = nz.sum(0)                                     # [ng]
    m = (Xn * nz.unsqueeze(-1)).sum(0) / nz_cnt.clamp_min(1).unsqueeze(-1)
    disp = (1.0 - m.norm(dim=1)).cpu()                     # [ng]
    disp[nz_cnt.cpu() == 0] = float("nan")

    npz = K.artifact(f"OC0_GRADIENT_ROWS_{scene}_{stage}.npz")
    part_np = np.asarray([0 if int(r) in hot_ids else 1 for r in sel_ids],
                         dtype=np.int8)
    np.savez_compressed(
        npz,
        X_preconditioned_fp32=X.numpy().astype(np.float32),
        row_norm_fp32=norms.numpy().astype(np.float32),
        selection_ids=np.asarray(sel_ids, dtype=np.int64),
        selection_partition=part_np,
        oc0_camera_ids=np.asarray(cams, dtype=np.int64),
        pair_a_slot=np.asarray([p[0] for p in pairs], dtype=np.int64),
        pair_b_slot=np.asarray([p[1] for p in pairs], dtype=np.int64),
        cos_fp32=cos_t.cpu().numpy().astype(np.float32),
        pair_valid=valid.cpu().numpy(),
        n_valid_pairs=n_valid_pair.cpu().numpy().astype(np.int64),
        negative_pair_fraction=neg_frac.numpy().astype(np.float64),
        severe_pair_fraction=sev_frac.numpy().astype(np.float64),
        directional_dispersion=disp.numpy().astype(np.float64),
        view_losses=np.asarray(losses, dtype=np.float64),
        row_dim=np.asarray(C.ROW_DIM),
        group_bounds=np.asarray([[a, b] for _, a, b in C.GROUPS], dtype=np.int64),
        group_names=np.asarray([n for n, _, _ in C.GROUPS]),
    )
    meta = {
        "campaign": "OPTLAWS0", "stage": "OC0_gradient_rows",
        "scene": scene, "checkpoint": stage,
        "n_cameras": nc, "n_gaussians": ng, "n_pairs": n_pair,
        "row_layout": ["xyz3", "scale3", "quat4", "opacity1", "sh_dc3", "sh_rest45"],
        "definitions": {
            "P_t": "x/(sqrt(v_t)+eps), per-optimizer-group eps (mg0_core.flat_denom)",
            "cosine_ab": "<P(g_a,j),P(g_b,j)>/(||.|| ||.||), defined only when both "
                         "row-gradient norms are nonzero",
            "Y_ab": "1 - cosine_ab",
            "conflict_ab": f"cosine_ab < {K.CONFLICT_COSINE}",
            "severe_ab": f"cosine_ab < {K.SEVERE_COSINE}",
            "negative_pair_fraction": "per-Gaussian mean over valid pairs of conflict_ab",
            "severe_pair_fraction": "per-Gaussian mean over valid pairs of severe_ab",
            "directional_dispersion": "1 - ||mean over nonzero-norm cameras of "
                                      "normalized gradient||; NaN if none",
        },
        "view_losses_mean": float(np.mean(losses)),
        "median_negative_pair_fraction_hot": float(np.median(
            neg_frac.numpy()[part_np == 0])),
        "median_negative_pair_fraction_uniform": float(np.median(
            neg_frac.numpy()[part_np == 1])),
        "artifact": os.path.basename(npz),
        "artifact_sha256": C.sha256_file(npz),
        "selection_ids_sha256": hashlib.sha256(
            np.asarray(sel_ids, dtype=np.int64).tobytes()).hexdigest(),
        "timing_s": time.time() - t0, "memory_gb_max": C.gpu_mem_gb(),
        "gpu_uuid": os.environ.get("MG0_GPU_UUID", "unknown"),
        "script_sha256": {os.path.basename(__file__):
                          C.sha256_file(os.path.abspath(__file__))},
    }
    C.json_dump(meta, K.artifact(f"OC0_GRADIENT_ROWS_{scene}_{stage}.json"))
    print(f"[oc0_gradient_rows] {scene}@{stage} COMPLETE {time.time() - t0:.0f}s "
          f"median_neg_frac={float(np.median(neg_frac.numpy())):.4f}", flush=True)


if __name__ == "__main__":
    main()