import json
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts import run_linux_tier_a_matrix as matrix  # noqa: E402
from scripts.run_linux_tier_a_matrix import (  # noqa: E402
    MATRIX_ORDER,
    _clean_checkout_commit,
    _validate_metric,
    _validate_report,
    build_plan,
    parse_nvml_process_memory,
    preflight_nvml,
    run_matrix,
    validate_case_assets,
    validate_session_metrics,
)


class LinuxTierAMatrixPlanTest(unittest.TestCase):
    def test_plan_contains_balanced_25_run_order(self):
        plan = build_plan(Path("/repo"), Path("/opt/miniforge/envs"))

        self.assertEqual(len(plan), 25)
        self.assertEqual(
            [(row["case_id"], row["renderer"]) for row in plan],
            list(MATRIX_ORDER),
        )
        self.assertEqual(
            [value.replace("\\", "/") for value in plan[0]["command"]],
            [
                "/opt/miniforge/envs/original3dgs/bin/python",
                "/repo/benchmark.py",
                "run",
                "original_3dgs",
                "--dataset",
                "small-garden-1080p",
            ],
        )
        self.assertEqual(
            plan[2]["command"][0].replace("\\", "/"),
            "/opt/miniforge/envs/gsplat/bin/python",
        )

    def test_nvml_parser_requires_numeric_memory_for_target_pid(self):
        output = "1234, python, 612 MiB\n4321, other, [N/A]\n"

        self.assertEqual(parse_nvml_process_memory(output, 1234), 612.0)
        with self.assertRaisesRegex(RuntimeError, "numeric NVML"):
            parse_nvml_process_memory(output, 4321)
        with self.assertRaisesRegex(RuntimeError, "not reported"):
            parse_nvml_process_memory(output, 9999)


class LinuxTierAMatrixEvidenceTest(unittest.TestCase):
    COMMIT = "1" * 40

    def _write_benchmark_metadata(self, root):
        benchmark = root / "benchmark"
        benchmark.mkdir(parents=True)
        protocol = {
            "protocol_id": "protocol-v2",
            "resolution": [1920, 1080],
            "color_space": "sRGB",
            "background": "black",
        }
        protocol_path = benchmark / "protocol.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        cases = []
        for case_id, _ in MATRIX_ORDER:
            if any(row["case_id"] == case_id for row in cases):
                continue
            dataset_id, scene_id = case_id.split("-", 1)[0], case_id
            cases.append({
                "case_id": case_id,
                "dataset_id": dataset_id,
                "scene_id": scene_id,
                "workload_tier": "test",
                "canonical_assets": {
                    "status": "pinned",
                    "checkpoint_sha256": "a" * 64,
                    "camera_sha256": "b" * 64,
                    "ground_truth_manifest_sha256": "c" * 64,
                },
            })
        suite = {
            "schema_version": "2.0",
            "suite_id": "suite",
            "version": "1.0",
            "primary_track": "track",
            "protocol_path": "benchmark/protocol.json",
            "protocol_sha256": protocol_sha,
            "cases": cases,
        }
        (benchmark / "suite.json").write_text(json.dumps(suite), encoding="utf-8")
        renderers = []
        for _, renderer in MATRIX_ORDER:
            if any(row["id"] == renderer for row in renderers):
                continue
            renderers.append({"id": renderer, "source_commit": "d" * 40})
        (benchmark / "renderers.json").write_text(
            json.dumps({"renderers": renderers}), encoding="utf-8"
        )
        return suite, protocol

    def test_case_asset_validator_recomputes_checkpoint_camera_and_gt_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_root = root / "datasets" / "processed" / "dataset" / "scene"
            image_root = case_root / "eval_images"
            image_root.mkdir(parents=True)
            scene = case_root / "point_cloud.ply"
            camera = case_root / "eval_cameras.json"
            image = image_root / "view.png"
            scene.write_bytes(b"scene")
            camera.write_bytes(b"camera")
            image.write_bytes(b"image")
            gt_entries = [{
                "image": "view.png",
                "sha256": hashlib.sha256(b"image").hexdigest(),
            }]
            gt_manifest = hashlib.sha256(json.dumps(
                gt_entries,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            preparation = {
                "status": "canonical",
                "ground_truth_files": gt_entries,
                "ground_truth_file_manifest_sha256": gt_manifest,
            }
            (case_root / "preparation.json").write_text(
                json.dumps(preparation), encoding="utf-8"
            )
            case = {
                "case_id": "case",
                "scene_path": str(scene.relative_to(root)),
                "camera_path": str(camera.relative_to(root)),
                "ground_truth_path": str(image_root.relative_to(root)),
                "preparation_path": str((case_root / "preparation.json").relative_to(root)),
                "canonical_assets": {
                    "status": "pinned",
                    "checkpoint_sha256": hashlib.sha256(b"scene").hexdigest(),
                    "camera_sha256": hashlib.sha256(b"camera").hexdigest(),
                    "ground_truth_manifest_sha256": gt_manifest,
                },
            }

            summary = validate_case_assets(root, case)

        self.assertEqual(summary["case_id"], "case")
        self.assertEqual(summary["ground_truth_file_count"], 1)

    def _write_metric(
        self, root, renderer, case_id, peak=512.0, gpu_uuid="GPU-1", run_name="run"
    ):
        suite = json.loads((root / "benchmark" / "suite.json").read_text())
        protocol = json.loads((root / "benchmark" / "protocol.json").read_text())
        case = next(row for row in suite["cases"] if row["case_id"] == case_id)
        run_dir = (
            root / "results" / "measured" / renderer /
            case["dataset_id"] / case["scene_id"] / run_name
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        raw_path = run_dir / "raw_samples.json"
        raw_path.write_text(json.dumps({
                "renderer_id": renderer,
                "case_id": case_id,
                "nvml_process_memory_samples": [
                    {"used_gpu_memory_mib": 0.0},
                    {"used_gpu_memory_mib": peak},
                ],
            }), encoding="utf-8")
        raw_uri = str(raw_path.relative_to(root)).replace("\\", "/")
        raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        metric_path = run_dir / "metrics.json"
        metric_path.write_text(
            json.dumps({
                "schema_version": "2.0",
                "result_id": f"{renderer}-{case_id}",
                "evidence_tier": "measured",
                "status": "complete",
                "renderer": {
                    "id": renderer, "config_id": renderer, "name": renderer,
                    "version": "1", "source_uri": "https://example.invalid",
                    "source_commit": "d" * 40, "build_command": "build",
                    "runtime_command": "run", "api": "CUDA", "backend": "CUDA",
                    "platforms": ["Linux"], "features": [],
                },
                "benchmark": {
                    "suite_id": suite["suite_id"], "suite_version": suite["version"],
                    "track_id": suite["primary_track"], "case_id": case_id,
                    "dataset_id": case["dataset_id"], "dataset_sha256": "e" * 64,
                    "scene_id": case["scene_id"], "scene_tier": "test",
                    "checkpoint_sha256": "a" * 64, "gaussian_count": 1,
                    "sh_degree": 3, "camera_trajectory_id": "eval",
                    "camera_trajectory_sha256": "b" * 64,
                    "quality_reference_sha256": "c" * 64,
                    "resolution": {"width": 1920, "height": 1080},
                    "color_space": "sRGB", "background": "black",
                    "protocol_id": protocol["protocol_id"],
                    "protocol_sha256": suite["protocol_sha256"],
                },
                "environment": {
                    "hardware_profile_id": "hardware", "gpu": "GPU",
                    "gpu_uuid": gpu_uuid,
                    "gpu_vram_mb": 24000, "cpu": "CPU", "ram_mb": 64000,
                    "driver": "600.1",
                    "cuda": "13.0",
                    "python": "3.12",
                    "pytorch": "2.12.1+cu130",
                    "benchmark_commit": self.COMMIT,
                    "os": "Linux",
                    "clock_policy": "default", "power_limit_w": 300,
                },
                "metrics": {
                    "performance": {
                        "fps": 10, "fps_ci95_low": 9, "fps_ci95_high": 11,
                        "frame_time_ms": 100, "p95_frame_time_ms": 110,
                        "p99_frame_time_ms": 120, "peak_vram_mb": peak,
                        "startup_time_ms": 1, "renderer_init_time_ms": 1,
                        "scene_load_time_ms": 1, "renderer_prepare_time_ms": 1,
                        "time_to_first_frame_ms": 1,
                    },
                    "quality": {"psnr_db": 30, "ssim": 0.9, "lpips": 0.1},
                    "raw_samples": {"uri": raw_uri, "sha256": raw_sha},
                },
                "provenance": {
                    "source_type": "repository_run", "source_uri": "run",
                    "measured_at": "2026-01-01T00:00:00Z",
                    "raw_samples_uri": raw_uri, "raw_samples_sha256": raw_sha,
                },
            }),
            encoding="utf-8",
        )
        return metric_path

    def test_session_validator_accepts_complete_positive_same_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_benchmark_metadata(root)
            paths = [
                self._write_metric(root, renderer, case_id)
                for case_id, renderer in MATRIX_ORDER
            ]

            summary = validate_session_metrics(paths, root, self.COMMIT)

        self.assertEqual(summary["metric_count"], 25)
        self.assertEqual(summary["renderer_count"], 5)
        self.assertEqual(summary["case_count"], 5)

    def test_session_validator_rejects_zero_nvml_and_mixed_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_benchmark_metadata(root)
            paths = [
                self._write_metric(root, renderer, case_id)
                for case_id, renderer in MATRIX_ORDER
            ]
            paths[0] = self._write_metric(
                root,
                MATRIX_ORDER[0][1],
                MATRIX_ORDER[0][0],
                peak=0.0,
                run_name="replacement",
            )
            with self.assertRaisesRegex(ValueError, "must be positive|positive NVML"):
                validate_session_metrics(paths, root, self.COMMIT)

            paths[0] = self._write_metric(
                root,
                MATRIX_ORDER[0][1],
                MATRIX_ORDER[0][0],
                gpu_uuid="GPU-2",
                run_name="replacement-two",
            )
            with self.assertRaisesRegex(ValueError, "single hardware cohort"):
                validate_session_metrics(paths, root, self.COMMIT)

    def test_metric_validator_enforces_registry_suite_and_raw_sample_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_benchmark_metadata(root)
            case_id, renderer = MATRIX_ORDER[0]
            metric_path = self._write_metric(root, renderer, case_id)

            _validate_metric(metric_path, root, self.COMMIT)

            document = json.loads(metric_path.read_text())
            document["renderer"]["source_commit"] = "f" * 40
            metric_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source commit"):
                _validate_metric(metric_path, root, self.COMMIT)

            document["renderer"]["source_commit"] = "d" * 40
            document["benchmark"]["checkpoint_sha256"] = "f" * 64
            metric_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                _validate_metric(metric_path, root, self.COMMIT)

            document["benchmark"]["checkpoint_sha256"] = "a" * 64
            document["provenance"]["raw_samples_sha256"] = "f" * 64
            metric_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hashes disagree"):
                _validate_metric(metric_path, root, self.COMMIT)

    def test_clean_checkout_commit_rejects_tracked_changes(self):
        with mock.patch.object(
            matrix.subprocess,
            "check_output",
            side_effect=[" M src/file.py\n"],
        ):
            with self.assertRaisesRegex(RuntimeError, "clean tracked checkout"):
                _clean_checkout_commit(Path("/repo"))

        with mock.patch.object(
            matrix.subprocess,
            "check_output",
            side_effect=["", self.COMMIT + "\n"],
        ):
            self.assertEqual(_clean_checkout_commit(Path("/repo")), self.COMMIT)

    def test_nvml_preflight_waits_for_holder_after_positive_live_sample(self):
        class Holder:
            def __init__(self):
                self.stdout = io.StringIO("1234\n")
                self.stderr = io.StringIO("")
                self.returncode = None
                self.wait_timeouts = []
                self.terminated = False

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                self.wait_timeouts.append(timeout)
                self.returncode = 0
                return 0

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.returncode = -9

        holder = Holder()
        query = SimpleNamespace(stdout="1234, python, 612 MiB\n")
        with mock.patch.object(matrix.subprocess, "Popen", return_value=holder) as popen, \
             mock.patch.object(matrix.subprocess, "run", return_value=query):
            memory = preflight_nvml(Path("/env/python"), holder_seconds=60.0)

        self.assertEqual(memory, 612.0)
        self.assertEqual(holder.wait_timeouts, [65.0])
        self.assertFalse(holder.terminated)
        self.assertIn("time.sleep(60.0)", popen.call_args.args[0][2])

    def test_report_requires_no_rejections_and_five_measured_renderers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranking.json"
            overall = [
                {"competitor_id": renderer}
                for renderer in sorted({renderer for _, renderer in MATRIX_ORDER})
            ]
            path.write_text(json.dumps({
                "rejected_files": [],
                "tiers": {"measured": {"overall": overall}},
            }), encoding="utf-8")
            _validate_report(path)

            path.write_text(json.dumps({
                "rejected_files": [{"source_file": "bad"}],
                "tiers": {"measured": {"overall": overall}},
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "rejected"):
                _validate_report(path)

    def test_resume_adopts_one_valid_orphan_without_rerunning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_benchmark_metadata(root)
            paths = [
                self._write_metric(root, renderer, case_id)
                for case_id, renderer in MATRIX_ORDER
            ]
            env_root = root / "envs"
            for environment in set(matrix.ENV_BY_RENDERER.values()):
                python = env_root / environment / "bin" / "python"
                python.parent.mkdir(parents=True)
                python.touch()
            session_path = root / "session.json"
            session_path.write_text(json.dumps({
                "schema_version": 1,
                "benchmark_commit": self.COMMIT,
                "completed": [{
                    "step": index,
                    "case_id": case_id,
                    "renderer": renderer,
                    "metrics_path": str(paths[index - 1].relative_to(root)),
                } for index, (case_id, renderer) in enumerate(MATRIX_ORDER[:-1], start=1)],
            }), encoding="utf-8")

            def fake_run(command, **kwargs):
                self.assertIn("report", command)
                report_dir = root / "docs" / "leaderboard"
                report_dir.mkdir(parents=True)
                report_dir.joinpath("ranking.json").write_text(json.dumps({
                    "rejected_files": [],
                    "tiers": {"measured": {"overall": [
                        {"competitor_id": renderer}
                        for renderer in sorted({row[1] for row in MATRIX_ORDER})
                    ]}},
                }), encoding="utf-8")
                return SimpleNamespace(returncode=0)

            with mock.patch.object(matrix.platform, "system", return_value="Linux"), \
                 mock.patch.object(matrix, "_clean_checkout_commit", return_value=self.COMMIT), \
                 mock.patch.object(matrix, "validate_canonical_assets", return_value=[]), \
                 mock.patch.object(matrix.subprocess, "run", side_effect=fake_run) as run:
                session = run_matrix(root, env_root, session_path, resume=True)

        self.assertEqual(len(session["completed"]), 25)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(session["report_output"].replace("\\", "/"), "docs/leaderboard")

    def test_resume_stops_on_multiple_orphans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_benchmark_metadata(root)
            case_id, renderer = MATRIX_ORDER[0]
            self._write_metric(root, renderer, case_id, run_name="one")
            self._write_metric(root, renderer, case_id, run_name="two")
            env_root = root / "envs"
            for environment in set(matrix.ENV_BY_RENDERER.values()):
                python = env_root / environment / "bin" / "python"
                python.parent.mkdir(parents=True)
                python.touch()
            session_path = root / "session.json"
            session_path.write_text(json.dumps({
                "schema_version": 1,
                "benchmark_commit": self.COMMIT,
                "completed": [],
            }), encoding="utf-8")
            with mock.patch.object(matrix.platform, "system", return_value="Linux"), \
                 mock.patch.object(matrix, "_clean_checkout_commit", return_value=self.COMMIT), \
                 mock.patch.object(matrix, "validate_canonical_assets", return_value=[]):
                with self.assertRaisesRegex(RuntimeError, "multiple unregistered metrics"):
                    run_matrix(root, env_root, session_path, resume=True)


if __name__ == "__main__":
    unittest.main()
