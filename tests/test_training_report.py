import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.generate_training_report import aggregate


class TrainingReportTest(unittest.TestCase):
    def test_incomplete_matrix_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                aggregate(root, Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
