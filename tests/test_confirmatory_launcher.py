"""Tests for the confirmatory matrix launcher and result collector."""

import json
import random
from pathlib import Path

import pytest

from src.scripts.run_confirmatory_matrix import (
    DEFAULT_SCENES,
    build_command,
    parse_scene_specs,
)
from src.scripts.collect_confirmatory_results import RUN_ID_RE, collect


def _cmd(arm="ctrl", coarse="0.5:0,1.0:1500"):
    return build_command(
        "mipnerf360/garden", arm, 3, 30000, 1920, 1080,
        "/data/processed", "/out/garden_ctrl_s3.json", coarse,
    )


def test_ctrl_command_has_no_decay_or_schedule():
    cmd = _cmd("ctrl")
    assert "--masked-adam-union-decay" not in cmd
    assert "--masked-adam-union-decay-eval-proj" not in cmd
    assert "--res-schedule" not in cmd
    assert "--masked-adam" in cmd
    assert "--steps" in cmd and cmd[cmd.index("--steps") + 1] == "30000"


def test_pd_command_has_frozen_quality_max_flags():
    cmd = _cmd("pd")
    assert "--masked-adam-union-decay" in cmd
    assert cmd[cmd.index("--masked-adam-union-decay") + 1] == "0.99"
    assert "--masked-adam-union-decay-eval-proj" in cmd
    assert "--res-schedule" in cmd
    assert cmd[cmd.index("--res-schedule") + 1] == "0.5:0,1.0:1500"



def test_pd_without_coarse_schedule_raises():
    import pytest

    with pytest.raises(ValueError, match="requires a coarse res-schedule"):
        build_command(
            "mipnerf360/garden", "pd", 3, 30000, 1920, 1080,
            "/data/processed", "/out/garden_pd_s3.json", None,
        )
def test_default_scenes_match_protocol():
    assert len(DEFAULT_SCENES) == 5
    assert ("mipnerf360/garden", "0.75:0,1.0:1500") in DEFAULT_SCENES
    assert ("tanks_and_temples/train", "0.5:0,1.0:1500") in DEFAULT_SCENES


def test_parse_scene_specs():
    specs = parse_scene_specs(["deep_blending/playroom:0.5:0,1.0:1500", "x/y"])
    assert specs == [("deep_blending/playroom", "0.5:0,1.0:1500"), ("x/y", None)]


def test_arm_order_is_deterministic_with_seed():
    def orders(seed):
        rng = random.Random(seed)
        return ["ctrl", "pd"] if rng.random() < 0.5 else ["pd", "ctrl"]

    assert orders(20260806) == orders(20260806)
    first = orders(20260806)
    assert set(first) == {"ctrl", "pd"}


def test_run_id_regex():
    match = RUN_ID_RE.match("mipnerf360_garden_pd_s4")
    assert match and match.group("scene_name") == "garden" and match.group("arm") == "pd"
    assert match.group("seed") == "4"


def _fake_run(dirpath: Path, run_id: str, psnr: float, curve: list):
    doc = {
        "device": "fake",
        "torch": "2.0",
        "config": {"steps": 30000},
        "scenes": {
            "mipnerf360/garden": [{
                "train_ms": 10.5, "total_wall_s": 123.0, "psnr": psnr,
                "ssim": 0.7, "lpips": 0.3, "final_n": 1000,
                "peak_vram_gb": 5.0, "eval_curve": curve,
            }]
        },
    }
    (dirpath / f"{run_id}.json").write_text(json.dumps(doc), encoding="utf-8")


def test_collect_builds_runs_and_time_to_target(tmp_path):
    _fake_run(tmp_path, "mipnerf360_garden_ctrl_s3", 20.0, [])
    _fake_run(
        tmp_path,
        "mipnerf360_garden_pd_s3",
        20.3,
        [{"step": 300, "wall_s": 5.0, "psnr": 19.5},
         {"step": 600, "wall_s": 10.0, "psnr": 20.1}],
    )
    summary = collect(tmp_path)
    assert summary["n_runs"] == 2
    assert summary["runs"]["mipnerf360_garden_ctrl_s3"]["arm"] == "ctrl"
    pd = summary["runs"]["mipnerf360_garden_pd_s3"]
    assert pd["train_ms"] == 10.5 and pd["total_wall_s"] == 123.0
    ttt = summary["time_to_target"]["mipnerf360_garden_pd_s3"]
    assert ttt["target_psnr"] == 20.0
    assert ttt["reached"] == {"step": 600, "wall_s": 10.0}


def test_collect_skips_errors_and_unmatched(tmp_path):
    (tmp_path / "garbage.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "other.json").write_text(
        json.dumps({"scenes": {"x": [{"error": "boom"}]}}), encoding="utf-8"
    )
    assert collect(tmp_path)["n_runs"] == 0
