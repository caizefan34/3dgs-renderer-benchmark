import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from public_links import PublicLinkError, validate_public_links


class PublicLinkTest(unittest.TestCase):
    def test_repository_public_entry_links_exist(self):
        summary = validate_public_links(
            ROOT,
            ["README.md", "docs/README.md", "paper/README.md", "CONTRIBUTING.md"],
        )
        self.assertGreater(summary["relative_links"], 20)

    def test_require_tracked_rejects_untracked_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("[tracked](tracked.md) [local](local.md)\n", encoding="utf-8")
            (root / "tracked.md").write_text("ok\n", encoding="utf-8")
            (root / "local.md").write_text("local\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "README.md", "tracked.md"], cwd=root, check=True)

            with self.assertRaisesRegex(PublicLinkError, "not Git-tracked.*local.md"):
                validate_public_links(root, ["README.md"], require_tracked=True)

    def test_rejects_missing_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("[missing](missing.md)\n", encoding="utf-8")
            with self.assertRaisesRegex(PublicLinkError, "missing target"):
                validate_public_links(root, ["README.md"])


if __name__ == "__main__":
    unittest.main()
