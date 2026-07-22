import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "scripts"))

from run_candidate_smoke_matrix import build_plan


class CandidateSmokeMatrixTest(unittest.TestCase):
    def test_plan_uses_three_isolated_environments(self):
        plan = build_plan(Path("/repo"), Path("/envs"), Path("/output"))
        self.assertEqual([row["renderer"] for row in plan], ["flashgs", "local_gs", "gemm_gs"])
        self.assertEqual([row["environment"] for row in plan], ["flashgs", "localgs", "gemmgs"])
        self.assertTrue(plan[0]["python"].replace("\\", "/").endswith("/envs/flashgs/bin/python"))


if __name__ == "__main__":
    unittest.main()
