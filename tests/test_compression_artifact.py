import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compression_artifact import decode_ply, encode_ply, read_binary_ply
from schema_validation import validate_schema


PROPERTIES = [
    "x", "y", "z", "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0", "f_rest_1", "f_rest_2",
    "opacity", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]


def _write_test_ply(path: Path, count: int = 48) -> np.ndarray:
    rng = np.random.default_rng(7)
    values = rng.normal(0.0, 0.2, (count, len(PROPERTIES))).astype("<f4")
    values[:, :3] += np.linspace(-2.0, 2.0, count)[:, None]
    values[:, 3:6] = 0.0
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header.extend(f"property float {name}" for name in PROPERTIES)
    header.append("end_header")
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(values.tobytes())
    return values


class CompressionArtifactTest(unittest.TestCase):
    def test_block_float_round_trip_and_manifest(self):
        self._round_trip("block-float", max_error=8e-5, block_size=16)

    def test_tile_codebook_round_trip_and_manifest(self):
        self._round_trip("tile-codebook", max_error=3e-3, tile_resolution=2)

    def _round_trip(self, codec: str, max_error: float, **options):
        root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (root / "benchmark" / "schemas" / "compression-artifact.schema.json").read_text()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "source.ply"
            archive = temp / f"scene.{codec}.zip"
            decoded = temp / "decoded.ply"
            manifest_path = temp / "manifest.json"
            expected = _write_test_ply(source)

            manifest = encode_ply(source, archive, codec, **options)
            decode_ply(archive, decoded)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            validate_schema(manifest, schema)

            layout, actual = read_binary_ply(decoded)
            matrix = np.column_stack([actual[name] for name in layout.properties])
            actual._mmap.close()
            if codec == "tile-codebook":
                matrix = matrix[np.argsort(matrix[:, 0])]
                expected = expected[np.argsort(expected[:, 0])]
            self.assertEqual(layout.properties, PROPERTIES)
            self.assertEqual(layout.count, len(expected))
            self.assertLessEqual(float(np.max(np.abs(matrix - expected))), max_error)
            self.assertEqual(manifest["codec"]["id"], codec)
            self.assertEqual(manifest["source"]["gaussian_count"], len(expected))
            self.assertEqual(manifest["compressed_artifact"]["sha256"], manifest["decode"]["artifact_sha256"])
            self.assertGreater(manifest["timings_ms"]["encode"], 0)
            self.assertGreater(manifest["timings_ms"]["decode_validation"], 0)


if __name__ == "__main__":
    unittest.main()
