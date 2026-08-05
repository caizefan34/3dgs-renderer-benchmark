import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "diagnostics" / "tcgs-variance-20260805"


class TcgsVarianceEvidenceTest(unittest.TestCase):
    def test_pass_structure_and_reported_tail_counts(self):
        expected = {
            "pass-a": {"warmup": 30, "stalls_over_100ms": 5},
            "pass-b": {"warmup": 150, "stalls_over_100ms": 1},
            "pass-c": {"warmup": 30, "stalls_over_100ms": 1},
        }
        quality_by_scene = {}

        for pass_name, contract in expected.items():
            artifacts = sorted(
                (EVIDENCE / pass_name).glob(
                    "**/tcgs/speed/benchmark_results.json"
                )
            )
            self.assertEqual(len(artifacts), 5)
            stalls = 0
            for artifact in artifacts:
                speed_document = json.loads(artifact.read_text(encoding="utf-8"))
                speed = speed_document["results"]["tcgs"]
                self.assertEqual(speed["warmup_frames"], contract["warmup"])
                stalls += speed["max_latency_ms"] > 100.0

                run_dir = artifact.parents[2]
                metrics = json.loads(
                    (run_dir / "metrics.json").read_text(encoding="utf-8")
                )
                scene = metrics["benchmark"]["scene_id"]
                psnr = metrics["metrics"]["quality"]["psnr_db"]
                quality_by_scene.setdefault(scene, set()).add(round(psnr, 9))

            self.assertEqual(stalls, contract["stalls_over_100ms"])

        self.assertEqual(set(quality_by_scene), {
            "bicycle", "bonsai", "garden", "train", "truck"
        })
        for values in quality_by_scene.values():
            self.assertEqual(len(values), 1)

    def test_diagnostics_are_outside_leaderboard_evidence_roots(self):
        relative = EVIDENCE.relative_to(ROOT / "results")
        self.assertEqual(relative.parts[0], "diagnostics")


if __name__ == "__main__":
    unittest.main()
