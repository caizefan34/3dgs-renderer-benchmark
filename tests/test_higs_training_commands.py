import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higs_training_commands import (  # noqa: E402
    HigsTrainingCommandError,
    audit_gsplat_source,
    build_training_invocation,
)


class HigsTrainingCommandTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "benchmark" / "higs-paper-protocol.json").read_text(
                encoding="utf-8"
            )
        )
    @staticmethod
    def _source(root: Path) -> tuple[Path, str]:
        source = root / "gsplat"
        trainer = source / "examples" / "simple_trainer.py"
        higs_api = (
            source
            / "gsplat"
            / "experimental"
            / "render"
            / "functional"
            / "gaussian_inference.py"
        )
        trainer.parent.mkdir(parents=True)
        higs_api.parent.mkdir(parents=True)
        trainer.write_text(
            'class Config:\n    init_type: str = "sfm"\n', encoding="utf-8"
        )
        higs_api.write_text(
            "def rasterize_gaussian_higs_dynamic():\n    pass\n", encoding="utf-8"
        )
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "config", "user.email", "test@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "config", "user.name", "Test"], check=True
        )
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(source), "commit", "-q", "-m", "fixture"], check=True
        )
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return source, commit

    def _protocol(self, commit: str) -> dict:
        protocol = copy.deepcopy(self.protocol)
        for method in ("gsplat", "higs_full", "higs_proposed"):
            protocol["methods"][method]["commit"] = commit
        return protocol

    def test_official_gsplat_command_is_from_sfm_and_exact_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, commit = self._source(Path(tmp))
            invocation = build_training_invocation(
                protocol=self._protocol(commit),
                method="gsplat",
                scene="mipnerf360/garden",
                seed=2,
                data_dir=Path(tmp) / "garden",
                result_dir=Path(tmp) / "result",
                source_dir=source,
                python_executable="python",
            )

        command = invocation["command"]
        self.assertEqual(invocation["initialization"], "from_scratch_sfm")
        self.assertEqual(invocation["iterations"], 30000)
        self.assertIn("--init-type", command)
        self.assertEqual(command[command.index("--scene") + 1], "mipnerf360/garden")
        self.assertEqual(command[command.index("--init-type") + 1], "sfm")
        self.assertEqual(command[command.index("--max-steps") + 1], "30000")
        self.assertNotIn("point_cloud.ply", " ".join(command).lower())
        self.assertEqual(invocation["environment"]["PYTHONPATH"], str(source.resolve()))
        self.assertIn("py", invocation["environment"]["TORCH_EXTENSIONS_DIR"])
        self.assertEqual(invocation["source"]["commit"], commit)

    def test_rejects_ply_or_checkpoint_as_training_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, commit = self._source(Path(tmp))
            for bad_name in ("point_cloud.ply", "scene.pt", "checkpoint.ckpt"):
                with self.subTest(bad_name=bad_name), self.assertRaisesRegex(
                    HigsTrainingCommandError, "dataset directory"
                ):
                    build_training_invocation(
                        protocol=self._protocol(commit),
                        method="gsplat",
                        scene="mipnerf360/garden",
                        seed=0,
                        data_dir=Path(bad_name),
                        result_dir=Path("result"),
                        source_dir=source,
                    )

    def test_higs_methods_fail_closed_without_densification_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, commit = self._source(Path(tmp))
            audit = audit_gsplat_source(source)
            self.assertTrue(audit["has_higs_dynamic_api"])
            self.assertFalse(audit["has_higs_densification_info"])

            with self.assertRaisesRegex(HigsTrainingCommandError, "densification"):
                build_training_invocation(
                    protocol=self._protocol(commit),
                    method="higs_full",
                    scene="mipnerf360/garden",
                    seed=0,
                    data_dir=Path("garden"),
                    result_dir=Path("result"),
                    source_dir=source,
                )

    def test_rejects_scene_seed_and_method_outside_frozen_protocol(self):
        cases = (
            {"method": "unknown", "scene": "mipnerf360/garden", "seed": 0},
            {"method": "gsplat", "scene": "private/foo", "seed": 0},
            {"method": "gsplat", "scene": "mipnerf360/garden", "seed": 99},
        )
        with tempfile.TemporaryDirectory() as tmp:
            source, commit = self._source(Path(tmp))
            for case in cases:
                with self.subTest(case=case), self.assertRaises(HigsTrainingCommandError):
                    build_training_invocation(
                        protocol=self._protocol(commit),
                        data_dir=Path("garden"),
                        result_dir=Path("result"),
                        source_dir=source,
                        **case,
                    )

    def test_repository_patch_is_lf_utf8_and_matches_frozen_protocol(self):
        patch = ROOT / "patches" / "higs-differentiable.patch"
        self.assertTrue(patch.is_file(), "missing patches/higs-differentiable.patch")
        raw = patch.read_bytes()
        self.assertFalse(raw.startswith(b"\xff\xfe"), "patch must not be UTF-16LE")
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "patch must not carry a BOM")
        self.assertEqual(raw.count(b"\r\n"), 0, "patch must be LF-only")
        digest = hashlib.sha256(raw).hexdigest()
        for method in ("higs_full", "higs_proposed"):
            spec = self.protocol["methods"][method]
            self.assertEqual(
                digest,
                spec["patch_sha256"],
                f"{method}.patch_sha256 must match the tracked patch bytes",
            )
            self.assertEqual(len(spec["source_diff_sha256"]), 64)
            self.assertEqual(len(spec["source_state_sha256"]), 64)
            self.assertEqual(len(spec["trainer_sha256"]), 64)

    def test_repository_higs_source_exposes_training_contract(self):
        candidates = [
            ROOT / "artifacts" / "renderer-sources" / "gsplat",
            ROOT / "artifacts" / "renderer-sources" / "gsplat-higs",
        ]
        sources = [path for path in candidates if path.is_dir()]
        self.assertTrue(sources, "no HiGS source checkout found under artifacts/renderer-sources")
        audit = audit_gsplat_source(sources[0])
        self.assertTrue(audit["has_higs_dynamic_api"])
        self.assertTrue(audit["has_higs_densification_info"])
        self.assertTrue(audit["has_higs_trainer_adapter"])


if __name__ == "__main__":
    unittest.main()
