import json
import tempfile
import unittest
from pathlib import Path


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.merge_compression_sessions import merge, summarize, write_report


class MergeCompressionSessionsTest(unittest.TestCase):
    def test_merge_and_aggregate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sessions = []
            for index, (case_id, source, compressed, psnr) in enumerate((
                ("a", 100, 20, -0.01), ("b", 200, 50, -0.03),
            )):
                metrics = root / f"metrics-{index}.json"
                metrics.write_text(json.dumps({
                    "case": {"case_id": case_id},
                    "codec": {"id": "codec", "artifact": {
                        "source_bytes": source, "compressed_bytes": compressed,
                    }},
                    "metrics": {
                        "quality_delta": {"psnr_db": psnr, "ssim": -0.001, "lpips": 0.002},
                        "near_lossless_gate": {"numeric_pass": True, "overall_pass": False},
                    },
                }), encoding="utf-8")
                session = root / f"session-{index}.json"
                session.write_text(json.dumps({
                    "benchmark_commit": "a" * 40, "status": "complete",
                    "completed": [{"case_id": case_id, "codec": "codec", "metrics_path": metrics.name}],
                }), encoding="utf-8")
                sessions.append(session)
            document = merge(sessions, root)
            rows = summarize(document)
            self.assertEqual(len(document["results"]), 2)
            self.assertAlmostEqual(rows[0]["compression_ratio"], 300 / 70)
            self.assertEqual(rows[0]["worst_psnr_delta_db"], -0.03)
            output = root / "report"
            write_report(document, output)
            self.assertIn("4.286x", (output / "compression-results.md").read_text())

    def test_merge_rejects_commit_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for index, commit in enumerate(("a" * 40, "b" * 40)):
                path = root / f"session-{index}.json"
                path.write_text(json.dumps({
                    "benchmark_commit": commit, "status": "complete", "completed": [],
                }), encoding="utf-8")
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "different benchmark commits"):
                merge(paths, root)


if __name__ == "__main__":
    unittest.main()
