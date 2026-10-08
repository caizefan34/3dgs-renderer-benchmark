"""Screen all eligible native split children by a full-resolution birth fit.

Only the first six training views select and fit a child. The other four
training views are inspected once after selection. This remains an offline
local proxy, not a complete trainer or untouched quality evaluation.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

import fastgs_birth_fit_probe as F


OUT = Path("/root/fastgs_birth_multi_candidate_result.json")


def main():
    started = time.perf_counter()
    P = F.P
    source = P.REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    if P.sha256(source) != "64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc":
        raise AssertionError("source PLY changed")
    mask = torch.load("/root/fastgs_gate_selected_4100_mask.pt", map_location="cpu",
                      weights_only=True)["selected_splits"]
    dataset = P.SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(P.REPO / "output/room_big_capture4100"),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=True)
    pipe = P.SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = P.GaussianModel(3, "default")
    scene = P.Scene(dataset, model, load_iteration=4100, shuffle=False)
    post, selected_ids, families = P.make_native_post(model, mask, seed=20260927)
    if families != 9345:
        raise AssertionError("native family count changed")
    with torch.no_grad():
        post._opacity[selected_ids] = model._opacity[selected_ids]
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    first_choice, selection, eligible = F.select_multiview_child(
        cameras=scene.getTrainCameras(), model=model, post=post,
        families=families, pipe=pipe)
    torch.cuda.synchronize()
    selection_seconds = time.perf_counter() - started - load_seconds
    records = []
    best_choice = None
    best_gain = -np.inf
    best_z = None
    for choice in eligible:
        child_id = len(model.get_xyz) + choice
        probe = P.append_child(model, post, child_id, F.Q_PROBE)
        fit_views = [F.collect(scene.getTrainCameras()[i], model, probe, pipe)
                     for i in F.CAMERA_INDICES[:6]]
        del probe
        valid = [v for v in fit_views if v["samples"]]
        samples = sum(v["samples"] for v in valid)
        record = {"choice": choice, "family": choice % families,
                  "fit_view_count": len(valid), "fit_pixels": samples,
                  "fit_per_view": [{"camera": v["camera"], "samples": v["samples"],
                                    "probe_parity_max": v.get("probe_parity_max")}
                                   for v in fit_views]}
        if samples >= 40 and len(valid) >= 2:
            a = np.concatenate([v["matrix"] for v in valid])
            y = np.concatenate([v["residual"] for v in valid])
            singular = np.linalg.svd(a, full_matrices=False)[1]
            record["singular_values"] = singular.tolist()
            if singular[-1] >= 1e-3:
                try:
                    z = F.fit_qp(a, y)
                except AssertionError as error:
                    record["fit_error"] = str(error)
                else:
                    before = float(y @ y)
                    after = float(np.sum((a @ z - y) ** 2))
                    gain = before - after
                    record.update({"opacity": float(z[3]),
                                   "rgb": (z[:3] / z[3]).tolist() if z[3] > 1e-8 else None,
                                   "fit_sse_before": before, "fit_sse_after": after,
                                   "fit_sse_gain": gain})
                    if gain > best_gain:
                        best_choice, best_gain, best_z = choice, gain, z
            else:
                record["fit_error"] = "rank deficient"
        else:
            record["fit_error"] = "insufficient stable fit support"
        records.append(record)
        print("FASTGS_BIRTH_CANDIDATE " + json.dumps(
            {k: record.get(k) for k in ("choice", "fit_pixels", "fit_view_count",
                                         "opacity", "fit_sse_gain", "fit_error")}), flush=True)

    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - started - load_seconds - selection_seconds
    output = {
        "source_ply_sha256": P.sha256(source),
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "camera_indices": F.CAMERA_INDICES,
        "first_choice_by_center_residual": first_choice,
        "selection": selection, "eligible_count": len(eligible),
        "eligibility_uses_validation_geometry_but_not_validation_gt": True,
        "fit_rule": "maximize absolute squared-error reduction on stable pixels in six fit training views, requiring >=40 pixels, >=2 views, and smallest design singular value >=1e-3",
        "candidates": records,
        "phase_seconds": {"load_and_native_proposal": load_seconds,
                          "visibility_selection": selection_seconds,
                          "candidate_fitting": fit_seconds},
        "status": "NO_STABLE_FIT" if best_choice is None else "SELECTED",
    }
    if best_choice is not None:
        child_id = len(model.get_xyz) + best_choice
        probe = P.append_child(model, post, child_id, F.Q_PROBE)
        held = [F.collect(scene.getTrainCameras()[i], model, probe, pipe)
                for i in F.CAMERA_INDICES[6:]]
        valid = [v for v in held if v["samples"]]
        output["selected"] = {
            "choice": best_choice, "family": best_choice % families,
            "parent_id": int(selected_ids[best_choice % families]),
            "opacity": float(best_z[3]),
            "rgb": (best_z[:3] / best_z[3]).tolist() if best_z[3] > 1e-8 else None,
            "fit_sse_gain": float(best_gain),
            "heldout_view_count": len(valid),
            "heldout_pixels": sum(v["samples"] for v in valid),
            "heldout_per_view": [{"camera": v["camera"], "samples": v["samples"],
                                  "probe_parity_max": v.get("probe_parity_max")}
                                 for v in held],
        }
        if valid:
            a = np.concatenate([v["matrix"] for v in valid])
            y = np.concatenate([v["residual"] for v in valid])
            before = float(y @ y)
            after = float(np.sum((a @ best_z - y) ** 2))
            output["selected"].update({
                "heldout_sse_before": before,
                "heldout_sse_after": after,
                "heldout_sse_gain": before - after})
    torch.cuda.synchronize()
    output["elapsed_seconds_including_scene_load"] = time.perf_counter() - started
    output["phase_seconds"]["heldout_and_serialization"] = (
        output["elapsed_seconds_including_scene_load"] - load_seconds
        - selection_seconds - fit_seconds)
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("FASTGS_BIRTH_MULTI_RESULT " + json.dumps({
        "status": output["status"], "eligible_count": len(eligible),
        "selected": output.get("selected"),
        "phase_seconds": output["phase_seconds"],
        "elapsed_seconds": output["elapsed_seconds_including_scene_load"]}), flush=True)


if __name__ == "__main__":
    main()
