"""Build and validate P12 paired-scene provenance manifests."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from p12_run_guard import SCENES, SCENE_ORDER, FUSED_COMMIT, FUSED_SO, GMAIN_SO, C0_SO


CAMPAIGNS = {"P12-A": ("GMAIN_FUSED", "C0_FUSED"), "P12-B": ("A0_FUSED", "A1_FUSED")}
RECIPE_KEYS = ("seed", "eps2d", "padding", "loss_weights", "absgrad", "absgrad_threshold", "iterations", "camera_sequence_sha256", "dataset_sha256", "optimizer_sha256", "schedule_sha256", "evaluation_sha256")
RUN_KEYS = ("scene", "arm", "gpu_uuid", "hostname", "start", "end", "repo_sha", "renderer_so_sha256", "fused_ssim_so_sha256", "command", "status", "recipe")


def _time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp lacks timezone: {value!r}")
    return parsed


def validate_pair(scene, runs, expected_order, arms):
    errors = []
    if len(runs) != 2:
        return [f"{scene}: expected exactly two runs, got {len(runs)}"]
    for run in runs:
        for key in RUN_KEYS:
            if key not in run or run[key] in (None, ""):
                errors.append(f"{scene}/{run.get('arm')}: missing {key}")
        recipe = run.get("recipe") or {}
        for key in RECIPE_KEYS:
            if key not in recipe or recipe[key] in (None, ""):
                errors.append(f"{scene}/{run.get('arm')}: missing recipe.{key}")
        if run.get("status") != "clean":
            errors.append(f"{scene}/{run.get('arm')}: contaminated or unverified status {run.get('status')!r}")
        if run.get("contaminated") is not False:
            errors.append(f"{scene}/{run.get('arm')}: contaminated flag must be explicitly false")
        if run.get("fused_ssim_so_sha256") != FUSED_SO:
            errors.append(f"{scene}/{run.get('arm')}: fused-loss binary differs from P11 frozen identity")
        if run.get("fused_ssim_source_commit") != FUSED_COMMIT:
            errors.append(f"{scene}/{run.get('arm')}: fused-loss source commit differs from P11 frozen identity")
        expected_so = GMAIN_SO if run.get("arm") == "GMAIN_FUSED" else C0_SO
        if run.get("renderer_so_sha256") != expected_so:
            errors.append(f"{scene}/{run.get('arm')}: renderer binary differs from frozen identity")
        for key, expected in {"seed": 42, "eps2d": 0.3, "padding": "same", "loss_weights": {"l1": 0.8, "ssim": 0.2}, "absgrad": True, "absgrad_threshold": 0.0008, "iterations": 30000}.items():
            if recipe.get(key) != expected:
                errors.append(f"{scene}/{run.get('arm')}: recipe.{key} expected {expected!r}, got {recipe.get(key)!r}")
        expected_mode = "classic" if run.get("arm") == "GMAIN_FUSED" else "higs_dynamic"
        if run.get("rasterization_mode") != expected_mode:
            errors.append(f"{scene}/{run.get('arm')}: rasterization mode expected {expected_mode!r}")
        if run.get("scene") != scene:
            errors.append(f"{scene}: run scene mismatch {run.get('scene')!r}")
    try:
        ordered = sorted(runs, key=lambda row: _time(row["start"]))
        if _time(ordered[0]["end"]) > _time(ordered[1]["start"]):
            errors.append(f"{scene}: paired runs overlap")
        for row in ordered:
            if _time(row["end"]) <= _time(row["start"]):
                errors.append(f"{scene}/{row['arm']}: end must be after start")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{scene}: invalid run timestamps: {exc}")
        ordered = runs
    order = [row.get("arm") for row in ordered]
    if set(order) != set(arms) or len(set(order)) != 2:
        errors.append(f"{scene}: expected arms {arms}, got {order}")
    if order != expected_order:
        errors.append(f"{scene}: arm order {order} differs from preregistered {expected_order}")
    if runs[0].get("gpu_uuid") != runs[1].get("gpu_uuid"):
        errors.append(f"{scene}: paired arms used different physical GPU UUIDs")
    if runs[0].get("hostname") != runs[1].get("hostname"):
        errors.append(f"{scene}: paired arms used different hosts")
    if runs[0].get("repo_sha") != runs[1].get("repo_sha"):
        errors.append(f"{scene}: paired arms used different repo SHAs")
    if any((runs[0].get("recipe") or {}).get(key) != (runs[1].get("recipe") or {}).get(key) for key in RECIPE_KEYS):
        errors.append(f"{scene}: paired arms used different recipes")
    if runs[0].get("fused_ssim_so_sha256") != runs[1].get("fused_ssim_so_sha256") or runs[0].get("fused_ssim_source_sha256") != runs[1].get("fused_ssim_source_sha256"):
        errors.append(f"{scene}: paired arms used different fused-loss identities")
    if not all(row.get("fused_ssim_source_sha256") or row.get("fused_ssim_source_commit") for row in runs):
        errors.append(f"{scene}: fused-loss source identity missing")
    if runs[0].get("fused_ssim_source_commit") != runs[1].get("fused_ssim_source_commit"):
        errors.append(f"{scene}: fused-loss source commits differ")
    return errors


def build_manifest(campaign, runs, preregistered_order):
    arms = CAMPAIGNS[campaign]
    errors, pairs = [], []
    first_order = preregistered_order.get(SCENE_ORDER[0])
    if first_order not in (list(arms), list(reversed(arms))):
        errors.append(f"{SCENE_ORDER[0]}: invalid first preregistered arm order {first_order!r}")
    for index, scene in enumerate(SCENE_ORDER):
        scene_runs = [row for row in runs if row.get("scene") == scene]
        order = preregistered_order.get(scene)
        if not order:
            errors.append(f"{scene}: missing preregistered arm order")
            order = []
        elif first_order in (list(arms), list(reversed(arms))) and order != (first_order if index % 2 == 0 else list(reversed(first_order))):
            errors.append(f"{scene}: preregistered order does not alternate across canonical scene order")
        errors.extend(validate_pair(scene, scene_runs, order, arms))
        if len(scene_runs) == 2:
            sorted_runs = sorted(scene_runs, key=lambda row: row.get("start", ""))
            pairs.append({"scene": scene, "arm_order": [row["arm"] for row in sorted_runs], "gpu_uuid": sorted_runs[0].get("gpu_uuid"), "hostname": sorted_runs[0].get("hostname"), "runs": sorted_runs, "clean_status": "clean" if all(row.get("status") == "clean" and row.get("contaminated") is False for row in sorted_runs) else "contaminated"})
    unknown = [row.get("scene") for row in runs if row.get("scene") not in SCENES]
    if unknown:
        errors.append(f"unknown scenes: {unknown}")
    if len(runs) != 26:
        errors.append(f"campaign: expected 26 runs, got {len(runs)}")
    return {"campaign": campaign, "status": "PASS" if not errors else "FAIL", "pairs": pairs, "violations": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=CAMPAIGNS, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--order-file", type=Path, required=True, help="preregistered JSON scene-to-arm-order map")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        runs = json.loads(args.runs.read_text(encoding="utf-8"))["runs"]
        order = json.loads(args.order_file.read_text(encoding="utf-8"))
        result = build_manifest(args.campaign, runs, order)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"pair manifest input: {exc}", file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for violation in result["violations"]:
        print(f"INVARIANT VIOLATION: {violation}", file=sys.stderr)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
