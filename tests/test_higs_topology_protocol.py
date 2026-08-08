"""Focused tests for the Phase-3 HiGS topology-preserving resolution protocol.

Covers the pre-registered exploration matrix (exploration_topology_11s0), the
protocol-driven patch resolution in ``build_training_invocation``, per-method
training budgets, and the audited trainer surface (resolution cache, window
schedule, scene calibration, segmented timing, full-resolution densification
anchors).
"""
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higs_ablation_protocol import (  # noqa: E402
    build_ablation_experiment_plan,
    validate_ablation_protocol,
)
from higs_training_commands import (  # noqa: E402
    HigsTrainingCommandError,
    _higs_method_ids,
    _method_iterations,
    audit_gsplat_source,
    build_training_invocation,
    trainer_cfg_kwargs,
)

TOPOLOGY_MATRIX = "exploration_topology_11s0"
TOPOLOGY_METHODS = ["gsplat_27k", "higs_visible_27k", "higs_topology_27k", "higs_topology_30k"]


class HigsTopologyProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-topology-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_topology_protocol_validates_as_ablation_protocol(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertEqual(report["planned_jobs"], 44)
        self.assertEqual(report["executable_jobs"], 44)
        self.assertEqual(report["confirmatory_jobs"], 0)

    def test_topology_matrix_has_four_methods_seed0_only(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == TOPOLOGY_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], TOPOLOGY_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_topology_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(plan), 44)
        from collections import Counter

        methods = Counter(job["method"] for job in plan)
        self.assertEqual(
            methods,
            {"gsplat_27k": 11, "higs_visible_27k": 11, "higs_topology_27k": 11, "higs_topology_30k": 11},
        )
        self.assertTrue(all(job["seed"] == 0 for job in plan))
        self.assertTrue(all(job["initialization"] == "from_scratch_sfm" for job in plan))

    def test_topology_window_schedule_contract(self):
        spec = self.protocol["methods"]["higs_topology_27k"]
        algo = spec["algorithm"]
        self.assertEqual(algo["max_steps"], 27000)
        schedule = algo["resolution_schedule"]
        self.assertEqual(schedule["mode"], "topology_window")
        self.assertEqual(schedule["lowres_start_step"], 15000)
        self.assertEqual(schedule["lowres_end_step"], 22000)
        self.assertEqual(schedule["scale"], 0.7)
        self.assertEqual(schedule["refine_stop_iter"], 15000)
        cfg = algo["trainer_cfg"]
        self.assertEqual(cfg["higs_lowres_start_step"], 15000)
        self.assertEqual(cfg["higs_lowres_end_step"], 22000)
        self.assertEqual(cfg["higs_train_res_scale"], 0.7)
        self.assertTrue(cfg["higs_res_cache"])
        self.assertTrue(cfg["higs_calibrate_scene"])
        self.assertEqual(cfg["higs_calibrate_min_speedup_ratio"], 0.85)
        self.assertTrue(cfg["higs_segment_timing"])
        self.assertTrue(cfg["higs_densify_anchor_fullres"])
        cfg30 = self.protocol["methods"]["higs_topology_30k"]["algorithm"]["trainer_cfg"]
        self.assertTrue(cfg30["higs_densify_anchor_fullres"])
        self.assertEqual(self.protocol["methods"]["higs_topology_30k"]["algorithm"]["max_steps"], 30000)

    def test_trainer_cfg_kwargs_passes_topology_fields(self):
        kwargs = trainer_cfg_kwargs(self.protocol["methods"]["higs_topology_27k"])
        self.assertFalse(kwargs["packed"])
        self.assertFalse(kwargs["sparse_grad"])
        self.assertTrue(kwargs["visible_adam"])
        self.assertEqual(kwargs["higs_lowres_start_step"], 15000)
        self.assertEqual(kwargs["higs_lowres_end_step"], 22000)
        self.assertEqual(kwargs["higs_train_res_scale"], 0.7)
        self.assertTrue(kwargs["higs_res_cache"])
        self.assertTrue(kwargs["higs_calibrate_scene"])
        self.assertTrue(kwargs["higs_segment_timing"])

    def test_method_iterations_respects_topology_budgets(self):
        methods = self.protocol["methods"]
        self.assertEqual(_method_iterations(self.protocol, methods["higs_topology_27k"]), 27000)
        self.assertEqual(_method_iterations(self.protocol, methods["higs_topology_30k"]), 30000)
        self.assertEqual(_method_iterations(self.protocol, methods["gsplat_27k"]), 27000)

    def test_topology_patch_is_lf_and_matches_protocol(self):
        patch = ROOT / "patches" / "higs-topology.patch"
        self.assertTrue(patch.is_file(), "missing patches/higs-topology.patch")
        raw = patch.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "patch must not carry a BOM")
        self.assertEqual(raw.count(b"\r\n"), 0, "patch must be LF-only")
        digest = hashlib.sha256(raw).hexdigest()
        for method_id in _higs_method_ids(self.protocol):
            spec = self.protocol["methods"][method_id]
            self.assertEqual(
                spec["patch_sha256"], digest,
                f"{method_id}.patch_sha256 must match patches/higs-topology.patch",
            )
            self.assertIn("patches/higs-topology.patch", spec["patches"])

    def test_topology_hashes_match_audited_source(self):
        source = ROOT / "artifacts" / "renderer-sources" / "gsplat-higs"
        if not source.is_dir():
            self.skipTest("patched gsplat-higs source checkout not present locally")
        audit = audit_gsplat_source(source)
        self.assertTrue(audit["has_higs_dynamic_api"])
        self.assertTrue(audit["has_higs_trainer_adapter"])
        self.assertTrue(audit["has_higs_densification_info"])
        for method_id in _higs_method_ids(self.protocol):
            spec = self.protocol["methods"][method_id]
            self.assertEqual(spec["trainer_sha256"], audit["trainer_sha256"])
            self.assertEqual(spec["source_state_sha256"], audit["source_state_sha256"])
            self.assertEqual(spec["source_diff_sha256"], audit["source_diff_sha256"])

    def test_build_invocation_resolves_protocol_patch_path(self):
        source = ROOT / "artifacts" / "renderer-sources" / "gsplat-higs"
        if not source.is_dir():
            self.skipTest("patched gsplat-higs source checkout not present locally")
        invocation = build_training_invocation(
            protocol=self.protocol,
            method="higs_topology_27k",
            scene="mipnerf360/garden",
            seed=0,
            data_dir=ROOT / "data" / "mipnerf360" / "garden",
            result_dir=ROOT / "artifacts" / "training-ablation" / "runs" / "test-topology-invocation",
            source_dir=source,
        )
        self.assertEqual(
            invocation["source"]["patch_sha256"],
            self.protocol["methods"]["higs_topology_27k"]["patch_sha256"],
        )
        self.assertEqual(invocation["iterations"], 27000)


class HigsTopologyTrainerSurfaceTest(unittest.TestCase):
    """Static surface checks on the audited patched trainer (no CUDA import)."""

    def setUp(self):
        self.trainer = ROOT / "artifacts" / "renderer-sources" / "gsplat-higs" / "examples" / "simple_trainer.py"
        if not self.trainer.is_file():
            self.skipTest("patched trainer not present locally")
        self.text = self.trainer.read_text(encoding="utf-8")

    def test_config_declares_topology_fields(self):
        for field in (
            "higs_lowres_start_step",
            "higs_lowres_end_step",
            "higs_res_cache",
            "higs_calibrate_scene",
            "higs_calibrate_min_speedup_ratio",
            "higs_calibrate_steps",
            "higs_densify_anchor_fullres",
            "higs_segment_timing",
            "higs_timing_start_step",
            "higs_timing_end_step",
        ):
            self.assertIn(field, self.text, f"Config field {field} missing")

    def test_trainer_exposes_phase3_helpers(self):
        for helper in (
            "def _effective_res_scale",
            "def _prepare_train_batch",
            "def _calibrate_resolution",
            "def _densify_anchor_pass",
            "def _timing_start",
            "def _timing_end",
            "def _schedule_has_lowres",
        ):
            self.assertIn(helper, self.text, f"helper {helper} missing")

    def test_trainer_writes_segmented_timing(self):
        self.assertIn("segmented_timing.json", self.text)
        self.assertIn("self._seg_times", self.text)
        self.assertIn("self._calibration_report", self.text)

    def test_trainer_caches_resolution_and_avoids_per_step_interpolate(self):
        self.assertIn("self._res_cache", self.text)
        self.assertIn('cache_key = (int(image_ids[0].item()), round(float(res_scale), 4))', self.text)

    def _calibration_slice(self):
        i = self.text.find("def _calibrate_resolution")
        j = self.text.find("def _densify_anchor_pass", i)
        self.assertGreater(j, i, "calibration region not found")
        return self.text[i:j]

    def test_calibration_times_forward_and_backward(self):
        cal = self._calibration_slice()
        self.assertIn("loss.backward()", cal)
        self.assertIn("end.record()", cal)
        self.assertLess(cal.index("loss.backward()"), cal.index("end.record()"))

    def test_calibration_warms_up_before_timing(self):
        cal = self._calibration_slice()
        self.assertIn("run_step(next_data())", cal)
        self.assertIn("timings.append(start.elapsed_time(end) / 1000.0)", cal)


if __name__ == "__main__":
    unittest.main()
