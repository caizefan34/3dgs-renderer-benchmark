"""Focused failure tests for the frozen P12 protocol guards."""

import copy
import math
from pathlib import Path
import statistics

import pytest

from p12_diff_guard import compare_trees
from p12_pair_manifest import RECIPE_KEYS, validate_pair
from p12_run_guard import SCENES, guard_run
from p12_stats_check import geometric_mean, log_ratio_ci, recompute, wall_reduction


def raw_campaign():
    runs = []
    for scene in sorted(SCENES):
        runs.extend([
            {"scene": scene, "arm": "A0_FUSED", "wall_time_s": 100.0, "final_n": 1000,
             "psnr": 30.0, "ssim": 0.9, "lpips": 0.1, "status": "clean", "contaminated": False},
            {"scene": scene, "arm": "A1_FUSED", "wall_time_s": 90.0, "final_n": 1100,
             "psnr": 30.1, "ssim": 0.91, "lpips": 0.09, "status": "clean", "contaminated": False},
        ])
    gm = 100 / 90
    return {"campaign": "P12-B", "runs": runs, "reported": {"gm": gm, "wall_reduction": wall_reduction(gm)}}


def test_geometric_mean_of_scene_ratios():
    assert geometric_mean([1, 4, 16]) == pytest.approx(4)
    assert recompute(raw_campaign())["gm"] == pytest.approx(100 / 90)


def test_wall_reduction_regression():
    assert wall_reduction(1.0979) * 100 == pytest.approx(8.9170234, abs=0.00001)
    assert wall_reduction(1.0979) * 100 != pytest.approx(9.79, abs=0.01)


def test_ci_student_t_on_log_ratios():
    ratios = [math.exp(i / 100) for i in range(-6, 7)]
    logs = [math.log(x) for x in ratios]
    half_width = 2.1788128296634177 * statistics.stdev(logs) / math.sqrt(13)
    assert log_ratio_ci(ratios) == pytest.approx([-half_width, half_width])
    assert "Student t" in recompute(raw_campaign())["ci95_method"]


def test_contaminated_attempt_excluded():
    raw = raw_campaign()
    dirty = copy.deepcopy(raw["runs"][0])
    dirty["wall_time_s"] = 1
    dirty["status"] = "contaminated"
    dirty["contaminated"] = True
    raw["runs"].append(dirty)
    result = recompute(raw)
    assert result["excluded_run_count"] == 1
    assert result["gm"] == pytest.approx(100 / 90)


def test_missing_scene_fails():
    raw = raw_campaign()
    raw["runs"] = [row for row in raw["runs"] if row["scene"] != "room"]
    with pytest.raises(ValueError, match="valid-scene count"):
        recompute(raw)


def test_wrong_reported_wall_reduction_fails():
    raw = raw_campaign()
    raw["reported"]["wall_reduction"] = raw["reported"]["gm"] - 1
    with pytest.raises(ValueError, match="1 - 1/GM"):
        recompute(raw)


def test_switch_invariant_failure_writes_json(tmp_path, capsys):
    config = {"arm": "A0_FUSED", "runtime_arm": "A0_FUSED", "scene": "room", "expected_scene": "room",
              "padding": "same", "eps2d": 0.3, "seed": 42, "loss_weights": {"l1": 0.8, "ssim": 0.2},
              "absgrad": True, "absgrad_threshold": 0.0008, "rasterization_mode": "higs_dynamic", "backward_mode": "higs_native",
              "iterations": 30000, "switches": {"HIGS_DISABLE_F9": "1", "HIGS_BWD_ABSGRAD": "1", "HIGS_PX_RUNTIME": "2"}}
    assert not guard_run(config, tmp_path, environ={"HIGS_DISABLE_F9": "0", "HIGS_BWD_ABSGRAD": "1", "HIGS_PX_RUNTIME": "2"})
    assert "switch HIGS_DISABLE_F9" in (tmp_path / "RUN_GUARD_FAIL.json").read_text()
    assert "INVARIANT VIOLATION" in capsys.readouterr().err


def test_pair_rejects_different_physical_gpus():
    recipe = {key: "x" for key in RECIPE_KEYS}
    recipe.update(seed=42, eps2d=0.3, padding="same", loss_weights={"l1": 0.8, "ssim": 0.2},
                  absgrad=True, absgrad_threshold=0.0008, iterations=30000)
    first = {"scene": "room", "arm": "A0_FUSED", "gpu_uuid": "GPU-one", "hostname": "host",
             "start": "2026-09-26T00:00:00Z", "end": "2026-09-26T01:00:00Z", "repo_sha": "abc",
             "renderer_so_sha256": "9baf8655f859f3e0f456e99a2d8e5b3ddfe414fe17a1a3e9d8072e0afed95b91",
             "fused_ssim_so_sha256": "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb",
             "fused_ssim_source_commit": "a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38", "command": "train",
             "status": "clean", "contaminated": False, "rasterization_mode": "higs_dynamic", "recipe": recipe}
    second = {**first, "arm": "A1_FUSED", "gpu_uuid": "GPU-two", "start": "2026-09-26T01:00:01Z", "end": "2026-09-26T02:00:00Z"}
    errors = validate_pair("room", [first, second], ["A0_FUSED", "A1_FUSED"], ("A0_FUSED", "A1_FUSED"))
    assert errors == ["room: paired arms used different physical GPU UUIDs"]


def test_unexpected_cuda_diff_fails(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "renderer.cu").write_text("old")
    (new / "renderer.cu").write_text("changed")
    result = compare_trees(old, new, "P8_C0_vs_P12_C0")
    assert result["changes"][0]["category"] == "unexpected"
    assert any("frozen historical renderer source differs" in error for error in result["violations"])


def test_a0_a1_unapproved_algorithmic_diff_fails(tmp_path):
    old, new = tmp_path / "a0", tmp_path / "a1"
    old.mkdir()
    new.mkdir()
    (old / "f9.cu").write_text("a")
    (new / "f9.cu").write_text("b")
    assert compare_trees(old, new, "A0_vs_A1")["violations"]
