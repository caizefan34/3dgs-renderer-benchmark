import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.run_linux_training_matrix import _ply_vertices, build_plan


class TrainingMatrixTest(unittest.TestCase):
    def test_repository_matrix_has_fifteen_fixed_budget_rows(self):
        root = Path(__file__).resolve().parents[1]
        rows = build_plan(root, Path("/repos"), Path("/envs"), Path("/outputs"))
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(row["iterations"] == 30000 for row in rows))
        self.assertTrue(all("--eval" in row["command"] for row in rows))

    def test_filters_select_one_row(self):
        root = Path(__file__).resolve().parents[1]
        rows = build_plan(root, Path("/repos"), Path("/envs"), Path("/outputs"),
                          {"gemm_gs_train"}, {"medium-train-1080p"})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["backend"]["commit"],
                         "aca61f897f58964ff7204e1e3c6485995b5f212c")

    def test_local_gs_records_cuda_compatibility_patch(self):
        root = Path(__file__).resolve().parents[1]
        rows = build_plan(root, Path("/repos"), Path("/envs"), Path("/outputs"),
                          {"local_gs_train"}, {"small-garden-1080p"})
        self.assertEqual(rows[0]["backend"]["patches"],
                         ["scripts/linux/patches/local-gs-simple-knn-cfloat.patch"])

    def test_reads_binary_ply_vertex_count_from_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.ply"
            path.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 42\nend_header\n")
            self.assertEqual(_ply_vertices(path), 42)


if __name__ == "__main__":
    unittest.main()
