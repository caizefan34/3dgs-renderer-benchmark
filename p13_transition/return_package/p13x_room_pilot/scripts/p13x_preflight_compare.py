#!/usr/bin/env python3
"""P13-X preflight comparison: baseline P12 trainer vs instrumented probe.

A (baseline, 220it) vs B (probe, diag OFF, 220it): loss checkpoints 1/100/200,
binary/renderer identity, camera sequence, checkpoint tensor equality.
C (probe, diag ON @100/150/200/250) vs D (probe, diag OFF, 300it): full
per-iteration loss trajectory, camera sequence hash, final param hash.
"""
import hashlib
import json
import os
import sys

import torch

R = "/root/epic05_p13x/runs"


def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def losses(res, iters=(1, 100, 200)):
    m = {r["iter"]: r["loss"] for r in res["per_iter"]}
    return {str(i): m.get(i) for i in iters}


def _cmp(va, vb):
    if torch.is_tensor(va) and torch.is_tensor(vb):
        if va.shape != vb.shape:
            return f"shape {tuple(va.shape)} vs {tuple(vb.shape)}"
        if torch.equal(va, vb):
            return "exact_equal"
        if not va.is_floating_point():
            return f"n_diff={int((va != vb).sum())}/{va.numel()}"
        return (f"max_abs_diff={(va.float()-vb.float()).abs().max().item():.3e} "
                f"n_diff={int((va != vb).sum())}/{va.numel()}")
    if isinstance(va, dict) and isinstance(vb, dict):
        return {k: _cmp(va[k], vb[k]) for k in sorted(set(va) & set(vb))} | (
            {k: "ONLY_A" for k in sorted(set(va) - set(vb))}
            | {k: "ONLY_B" for k in sorted(set(vb) - set(va))})
    if isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
        if len(va) != len(vb):
            return f"len {len(va)} vs {len(vb)}"
        return [_cmp(x, y) for x, y in zip(va, vb)]
    if isinstance(va, (int, float, str, bool)) or va is None:
        return "exact_equal" if va == vb else f"{va!r} != {vb!r}"
    return "exact_equal" if va is vb else f"type {type(va)} uncompared"


def compare_ckpt(pa, pb):
    a = torch.load(pa, map_location="cpu", weights_only=False)
    b = torch.load(pb, map_location="cpu", weights_only=False)
    return {"top_level": _cmp(a, b)}


def main():
    A = json.load(open(f"{R}/preflight_A_baseline220/results.json"))
    B = json.load(open(f"{R}/preflight_B_probe220/results.json"))
    C = json.load(open(f"{R}/preflight_C_diag300/results.json"))
    D = json.load(open(f"{R}/preflight_D_probe300/results.json"))
    lgB = json.load(open(f"{R}/preflight_B_probe220/p13x_iter_log.json"))
    lgC = json.load(open(f"{R}/preflight_C_diag300/p13x_iter_log.json"))
    lgD = json.load(open(f"{R}/preflight_D_probe300/p13x_iter_log.json"))

    def camsha(run):
        return sha256f(f"{R}/{run}/camera_sequence.npy")

    rep = {"phase": "P13-X preflight (paired replay)"}
    rep["A_vs_B"] = {
        "loss_1_100_200_A": losses(A), "loss_1_100_200_B": losses(B),
        "loss_deltas": {k: (None if losses(A)[k] is None or losses(B)[k] is None
                            else abs(losses(A)[k] - losses(B)[k]))
                        for k in losses(A)},
        "binary_identity_A": A["binary_identity"].get("so_sha256"),
        "binary_identity_B": B["binary_identity"].get("so_sha256"),
        "renderer_A": A["renderer"], "renderer_B": B["renderer"],
        "camera_sequence_sha256_A": camsha("preflight_A_baseline220"),
        "camera_sequence_sha256_B": camsha("preflight_B_probe220"),
        "final_N_A": A["final_N"], "final_N_B": B["final_N"],
        "pearson_ckpt_A_sha": sha256f(f"{R}/preflight_A_baseline220/ckpt_220.pt"),
        "pearson_ckpt_B_sha": sha256f(f"{R}/preflight_B_probe220/ckpt_220.pt"),
        "ckpt_compare": compare_ckpt(f"{R}/preflight_A_baseline220/ckpt_220.pt",
                                     f"{R}/preflight_B_probe220/ckpt_220.pt"),
    }
    lc = {i: (lv, n) for i, lv, n in lgC["iter_log"]}
    ld = {i: (lv, n) for i, lv, n in lgD["iter_log"]}
    common = sorted(set(lc) & set(ld))
    ndiff = sum(1 for i in common if lc[i][0] != ld[i][0])
    mx = max((abs(lc[i][0] - ld[i][0]) for i in common), default=None)
    first = next((i for i in common if lc[i][0] != ld[i][0]), None)
    lcB = {i: (lv, n) for i, lv, n in lgB["iter_log"]}
    la = {r["iter"]: r["loss"] for r in A["per_iter"]}
    rep["C_vs_D"] = {
        "n_common_iters": len(common),
        "n_loss_exact_equal": len(common) - ndiff,
        "n_loss_differing": ndiff,
        "max_abs_loss_diff": mx,
        "first_differing_iter": first,
        "camera_sequence_sha256_C": lgC["camera_sequence_sha256"],
        "camera_sequence_sha256_D": lgD["camera_sequence_sha256"],
        "final_model_param_sha256_C": lgC["final_model_param_sha256"],
        "final_model_param_sha256_D": lgD["final_model_param_sha256"],
        "final_params_equal": lgC["final_model_param_sha256"]
        == lgD["final_model_param_sha256"],
        "final_N_C": lgC["final_N"], "final_N_D": lgD["final_N"],
        "loss_1_100_200_C": {str(i): lc.get(i, (None,))[0] for i in (1, 100, 200)},
        "loss_1_100_200_D": {str(i): ld.get(i, (None,))[0] for i in (1, 100, 200)},
        "B_vs_D_traj_checksum_match":
            lgB["camera_sequence_sha256"] == lgD["camera_sequence_sha256"],
    }
    rep["diag_records"] = []
    dd = "/root/epic05_p13x/artifacts/preflight_snapshots"
    for fn in sorted(os.listdir(dd)):
        if fn.startswith("p13x_diag_record_"):
            r = json.load(open(os.path.join(dd, fn)))
            rep["diag_records"].append({
                "file": fn, "iteration": r["iteration"],
                "rng_restored_exactly": r["rng_restored_exactly"],
                "model_params_unchanged": r["model_params_unchanged"],
                "n_anchors": len(r["anchors"]),
                "anchor0_visible": r["anchors"][0]["n_visible_intersect_all"],
                "anchor0_mixed_axis": r["anchors"][0]["n_mixed_axis_radii"],
                "wall_ms": r["wall_ms"],
            })

    # D vs D2: identical command, diag OFF in both -- intrinsic noise floor
    p2 = f"{R}/preflight_D2_probe300/p13x_iter_log.json"
    if os.path.exists(p2):
        lgD2 = json.load(open(p2))
        l2 = {i: (lv, n) for i, lv, n in lgD2["iter_log"]}
        common2 = sorted(set(ld) & set(l2))
        nd2 = sum(1 for i in common2 if ld[i][0] != l2[i][0])
        rep["D_vs_D2_noise_floor"] = {
            "n_common_iters": len(common2),
            "n_loss_exact_equal": len(common2) - nd2,
            "n_loss_differing": nd2,
            "max_abs_loss_diff": max((abs(ld[i][0] - l2[i][0]) for i in common2),
                                     default=None),
            "first_differing_iter": next(
                (i for i in common2 if ld[i][0] != l2[i][0]), None),
            "final_params_equal": lgD["final_model_param_sha256"]
            == lgD2["final_model_param_sha256"],
            "camera_sequence_sha256_D2": lgD2["camera_sequence_sha256"],
            "loss_1_100_200_D2": {str(i): l2.get(i, (None,))[0] for i in (1, 100, 200)},
        }
    out = "/root/epic05_p13x/artifacts/P13X_PREFLIGHT_COMPARE.json"
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))
    print("WROTE", out)


if __name__ == "__main__":
    main()
