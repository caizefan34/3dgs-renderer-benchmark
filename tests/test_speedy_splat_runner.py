import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higs_training_commands import (  # noqa: E402
    HigsTrainingCommandError,
    audit_speedy_splat_source,
    build_speedy_splat_invocation,
)
from scripts.assemble_original_3dgs_results import assemble  # noqa: E402
from scripts.run_speedy_splat_a100_matrix import _plan_jobs  # noqa: E402


def _git(source: Path) -> str:
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "fixture"], check=True)
    return subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source(root: Path) -> tuple[Path, str]:
    source = root / "speedy_splat"
    (source / "utils").mkdir(parents=True)
    (source / "submodules" / "diff-gaussian-rasterization").mkdir(parents=True)
    (source / "train.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--seed", type=int, default=0)\n'
        "safe_state(args.quiet, args.seed)\n"
        "network_gui.init(args.ip, args.port)\n",
        encoding="utf-8",
    )
    (source / "utils" / "general_utils.py").write_text(
        "def safe_state(silent, seed=0):\n    pass\n",
        encoding="utf-8",
    )
    (source / "submodules" / "diff-gaussian-rasterization" / "setup.py").write_text(
        "from setuptools import setup\nsetup(name='diff_gaussian_rasterization')\n",
        encoding="utf-8",
    )
    return source, _git(source)


class SpeedySplatCommandTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "benchmark" / "higs-paper-protocol.json").read_text(encoding="utf-8")
        )

    def _protocol(self, source: Path) -> dict:
        protocol = copy.deepcopy(self.protocol)
        audit = audit_speedy_splat_source(source)
        method = protocol["methods"]["speedy_splat"]
        method["commit"] = audit["head_commit"]
        method["trainer_sha256"] = audit["trainer_sha256"]
        method["source_diff_sha256"] = audit["source_diff_sha256"]
        method["source_state_sha256"] = audit["source_state_sha256"]
        return protocol

    def test_audit_detects_seed_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = _source(Path(tmp))
            audit = audit_speedy_splat_source(source)
        self.assertTrue(audit["git_root_verified"])
        self.assertTrue(audit["has_train_py"])
        self.assertTrue(audit["has_seed_argument"])
        self.assertTrue(audit["uses_seeded_safe_state"])
        self.assertTrue(audit["safe_state_seeded"])
        self.assertEqual(len(audit["trainer_sha256"]), 64)

    def test_audit_fails_closed_without_seed_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "speedy_splat"
            (source / "utils").mkdir(parents=True)
            (source / "train.py").write_text(
                "safe_state(args.quiet)\n", encoding="utf-8"
            )
            (source / "utils" / "general_utils.py").write_text(
                "def safe_state(silent):\n    pass\n", encoding="utf-8"
            )
            _git(source)
            audit = audit_speedy_splat_source(source)
        self.assertFalse(audit["has_seed_argument"])
        self.assertFalse(audit["uses_seeded_safe_state"])

    def test_build_command_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = _source(Path(tmp))
            invocation = build_speedy_splat_invocation(
                protocol=self._protocol(source),
                scene="mipnerf360/garden",
                seed=2,
                data_dir=Path(tmp) / "garden",
                result_dir=Path(tmp) / "result",
                source_dir=source,
                python_executable="python",
            )
        command = invocation["command"]
        self.assertEqual(invocation["method"], "speedy_splat")
        self.assertEqual(invocation["iterations"], 30000)
        self.assertEqual(invocation["eval_steps"], [7000, 15000, 30000])
        self.assertEqual(command[command.index("-i") + 1], "images_4")
        # -u is required: the official trainer must flush the "[ITER N] Saving
        # Gaussians" markers immediately so the runner can timestamp them.
        self.assertEqual(command[command.index("-u") + 1].endswith("train.py"), True)
        self.assertEqual(command[command.index("--iterations") + 1], "30000")
        self.assertIn("--resolution=-1", command)
        self.assertEqual(command[command.index("--seed") + 1], "2")
        self.assertIn("--eval", command)
        # --quiet must NOT be passed: official safe_state(silent=True) swallows
        # stdout, hiding the "[ITER N] Saving Gaussians" markers the runner needs.
        self.assertNotIn("--quiet", command)
        # Speedy-Splat train.py has no --disable_viewer; its network_gui.init()
        # binds a localhost server, so concurrent jobs pass a unique --port.
        self.assertNotIn("--disable_viewer", command)

    def test_build_command_sets_unique_gui_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = _source(Path(tmp))
            invocation = build_speedy_splat_invocation(
                protocol=self._protocol(source),
                scene="mipnerf360/garden",
                seed=0,
                data_dir=Path(tmp) / "garden",
                result_dir=Path(tmp) / "result",
                source_dir=source,
                python_executable="python",
                gui_port=6016,
            )
        command = invocation["command"]
        self.assertIn("--port", command)
        self.assertEqual(command[command.index("--port") + 1], "6016")

    def test_build_command_default_has_no_port_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = _source(Path(tmp))
            invocation = build_speedy_splat_invocation(
                protocol=self._protocol(source),
                scene="mipnerf360/garden",
                seed=0,
                data_dir=Path(tmp) / "garden",
                result_dir=Path(tmp) / "result",
                source_dir=source,
                python_executable="python",
            )
        self.assertNotIn("--port", invocation["command"])

    def test_smoke_uses_single_eval_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = _source(Path(tmp))
            invocation = build_speedy_splat_invocation(
                protocol=self._protocol(source),
                scene="tanks_and_temples/train",
                seed=0,
                data_dir=Path(tmp) / "train",
                result_dir=Path(tmp) / "result",
                source_dir=source,
                smoke_steps=700,
            )
        self.assertEqual(invocation["iterations"], 700)
        self.assertEqual(invocation["eval_steps"], [700])
        self.assertEqual(
            invocation["command"][invocation["command"].index("--iterations") + 1],
            "700",
        )

    def test_rejects_ply_as_training_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = _source(Path(tmp))
            with self.assertRaisesRegex(HigsTrainingCommandError, "dataset directory"):
                build_speedy_splat_invocation(
                    protocol=self._protocol(source),
                    scene="mipnerf360/garden",
                    seed=0,
                    data_dir=Path("point_cloud.ply"),
                    result_dir=Path("result"),
                    source_dir=source,
                )

    def test_scheduler_plan_is_33_speedy_jobs(self):
        protocol = json.loads(
            (ROOT / "benchmark" / "higs-paper-protocol.json").read_text(encoding="utf-8")
        )
        # The committed protocol keeps speedy_splat blocked until smoke
        # evidence lands; the scheduler plan test asserts the ready-state
        # contract so it is independent of external evidence status.
        protocol["methods"]["speedy_splat"]["runner_status"] = "ready"
        jobs = _plan_jobs(protocol)
        self.assertEqual(len(jobs), 33)
        for job in jobs:
            self.assertEqual(job["method"], "speedy_splat")
            self.assertEqual(job["hardware"], "a100")
            self.assertTrue(job["executable"])
        scenes = {job["scene"] for job in jobs}
        self.assertEqual(len(scenes), 11)
        seeds = {job["seed"] for job in jobs}
        self.assertEqual(seeds, {0, 1, 2})


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "stats").mkdir(parents=True)
    for step, wall in ((6999, 300.0), (14999, 900.0), (29999, 2400.0)):
        (run / "stats" / f"train_step{step:04d}_rank0.json").write_text(
            json.dumps({"mem": 3.2, "ellipse_time": wall, "num_GS": 120000}),
            encoding="utf-8",
        )
    for step, psnr in ((6999, 22.0), (14999, 24.5), (29999, 26.0)):
        (run / "stats" / f"val_step{step:04d}.json").write_text(
            json.dumps({"psnr": psnr, "ssim": 0.88, "lpips": 0.12, "num_GS": 130000}),
            encoding="utf-8",
        )
    final_ply = run / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    final_ply.parent.mkdir(parents=True)
    final_ply.write_bytes(b"ply-bytes" * 1000)
    (run / "paper-run-metadata.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "method": "speedy_splat",
            "scene": "mipnerf360/garden",
            "seed": 1,
            "wall_time_seconds": 2500.0,
            "started_at_utc": "2026-08-07T00:00:00+00:00",
            "hardware": {"gpu_name": "NVIDIA A100-SXM4-80GB"},
            "dataset": {"inventory_sha256": "b" * 64},
            "source": {"commit": "54c035f7834b564019656c3e3fcc3646292f727d"},
            "resources": {"energy_joules": 500000.0, "peak_gpu_memory_mib": 4096.0},
        }),
        encoding="utf-8",
    )
    return run


JOB = {
    "job_id": "primary_full_convergence--speedy_splat--mipnerf360-garden--a100--s1",
    "method": "speedy_splat",
    "scene": "mipnerf360/garden",
    "hardware": "a100",
    "seed": 1,
    "initialization": "from_scratch_sfm",
    "iterations": 30000,
}


def test_assemble_speedy_splat(tmp_path):
    run = _make_run(tmp_path)
    result = assemble(run, JOB, gpu_index=2, energy_joules=0.0)
    assert result["status"] == "complete"
    assert result["job_id"] == JOB["job_id"]
    assert result["quality"]["psnr_db"] == 26.0
    assert result["performance"]["wall_time_seconds"] == 2500.0
    assert [p["iteration"] for p in result["quality_curve"]] == [7000, 15000, 30000]
    walls = [p["wall_time_seconds"] for p in result["quality_curve"]]
    assert walls == sorted(walls) and len(set(walls)) == 3
    assert result["resources"]["peak_gpu_memory_mib"] == 4096.0
    assert result["resources"]["energy_joules"] == 500000.0
    assert result["resources"]["final_gaussian_count"] == 130000
    assert result["artifact"]["path"].endswith("point_cloud.ply")
    assert len(result["artifact"]["sha256"]) == 64
    assert result["performance"]["time_to_quality_seconds"] <= 2500.0


def test_assemble_rejects_missing_final_ply(tmp_path):
    run = _make_run(tmp_path)
    (run / "point_cloud" / "iteration_30000" / "point_cloud.ply").unlink()
    with pytest.raises(ValueError, match="final checkpoint PLY"):
        assemble(run, JOB, gpu_index=2, energy_joules=0.0)
