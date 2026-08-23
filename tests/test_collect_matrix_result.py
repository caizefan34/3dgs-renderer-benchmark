import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.collect_matrix_result import _verify_render_outputs


class RenderOutputEvidenceTest(unittest.TestCase):
    def test_verifier_rejects_tampered_render_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "renderer" / "000-view.png"
            output.parent.mkdir()
            Image.new("RGB", (1, 1), (1, 2, 3)).save(output, format="PNG")
            frames = [{
                "frame": 0,
                "image": "view.png",
                "render_output": {
                    "path": "renderer/000-view.png",
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "format": "png",
                    "source_tensor_dtype": "float32",
                    "source_tensor_shape": [1, 1, 3],
                    "export_encoding": "RGB8 PNG",
                },
            }]
            document = {"render_outputs": {"root": str(root)}}
            self.assertEqual(len(_verify_render_outputs(document, frames)), 1)
            output.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _verify_render_outputs(document, frames)


if __name__ == "__main__":
    unittest.main()
