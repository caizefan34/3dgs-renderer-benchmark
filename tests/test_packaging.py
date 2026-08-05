import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackagingContractTest(unittest.TestCase):
    def _build_wheel(self, temp_dir: str) -> Path:
        environment = os.environ.copy()
        environment["PIP_CACHE_DIR"] = str(Path(temp_dir) / "pip-cache")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                temp_dir,
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return next(Path(temp_dir).glob("*.whl"))

    def test_wheel_contains_runtime_packages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._build_wheel(temp_dir)
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())

        for required in (
            "benchmark_cli.py",
            "benchmark_entrypoint.py",
            "benchmark_matrix.py",
            "benchmark_framework/__init__.py",
            "renderers/__init__.py",
            "scripts/run_local_renderer_suite.py",
        ):
            self.assertIn(required, names)

    def test_wheel_entrypoint_runs_outside_checkout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._build_wheel(temp_dir)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(wheel)
            environment["GSBENCH_ROOT"] = str(ROOT)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from benchmark_entrypoint import main; "
                    "raise SystemExit(main(['list', 'datasets']))",
                ],
                cwd=temp_dir,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("small-garden-1080p", completed.stdout)


if __name__ == "__main__":
    unittest.main()
