import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_entrypoint import find_repository_root


class BenchmarkEntrypointTest(unittest.TestCase):
    def test_explicit_repository_root_is_honored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(os.environ, {"GSBENCH_ROOT": str(ROOT)}):
                self.assertEqual(find_repository_root(Path(temp_dir)), ROOT)

    def test_invalid_explicit_repository_root_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(os.environ, {"GSBENCH_ROOT": temp_dir}):
                with self.assertRaisesRegex(SystemExit, "not a .* checkout"):
                    find_repository_root()

    def test_parent_checkout_is_discovered(self):
        self.assertEqual(find_repository_root(ROOT / "docs"), ROOT)


if __name__ == "__main__":
    unittest.main()
