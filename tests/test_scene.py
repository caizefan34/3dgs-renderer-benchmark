import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_framework.scene import _extract_sh_coefficients


class ExtractShCoefficientsTest(unittest.TestCase):
    def test_standard_3dgs_layout(self):
        records = {
            "f_dc_0": np.array([1.0], dtype=np.float32),
            "f_dc_1": np.array([2.0], dtype=np.float32),
            "f_dc_2": np.array([3.0], dtype=np.float32),
        }
        for index in range(9):
            records[f"f_rest_{index}"] = np.array([10.0 + index], dtype=np.float32)

        shs = _extract_sh_coefficients(records, list(records))

        self.assertEqual(shs.shape, (1, 4, 3))
        np.testing.assert_array_equal(shs[0, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(shs[0, 1], [10.0, 13.0, 16.0])
        np.testing.assert_array_equal(shs[0, 3], [12.0, 15.0, 18.0])

    def test_legacy_all_f_dc_layout(self):
        records = {
            f"f_dc_{index}": np.array([float(index)], dtype=np.float32)
            for index in range(12)
        }

        shs = _extract_sh_coefficients(records, list(records))

        self.assertEqual(shs.shape, (1, 4, 3))
        np.testing.assert_array_equal(shs[0, 0], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(shs[0, 3], [9.0, 10.0, 11.0])


if __name__ == "__main__":
    unittest.main()
