import base64
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from prepare_datasets import _crc32c, _safe_extract_zip, prepare


def _checksum_source(archive: Path) -> dict:
    payload = archive.read_bytes()
    return {
        "url": None,
        "size_bytes": len(payload),
        "checksums": {
            "md5_base64": base64.b64encode(hashlib.md5(payload).digest()).decode(),
            "crc32c_base64": base64.b64encode(_crc32c(payload).to_bytes(4, "big")).decode(),
        },
        "archive_format": "zip",
    }


class DatasetPipelineTest(unittest.TestCase):
    def test_official_manifests_pin_transport_identities(self):
        mip = json.loads((ROOT / "benchmark" / "datasets" / "mipnerf360.json").read_text(encoding="utf-8"))
        tandt = json.loads((ROOT / "benchmark" / "datasets" / "tanks_and_temples.json").read_text(encoding="utf-8"))
        for scene in ("garden", "bicycle", "bonsai"):
            source = mip["scenes"][scene]["source"]
            self.assertGreater(source["size_bytes"], 1_000_000_000)
            self.assertTrue(source["generation"].isdigit())
            self.assertTrue(source["checksums"]["md5_base64"])
            self.assertTrue(source["checksums"]["crc32c_base64"])
        for scene in ("truck", "train"):
            source = tandt["scenes"][scene]["source"]
            self.assertEqual(source["size_bytes"], 682628995)
            self.assertEqual(
                source["checksums"]["sha256"],
                "816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1",
            )
        for manifest in (mip, tandt):
            model = manifest["official_model_archive"]
            self.assertEqual(model["size_bytes"], 14660630999)
            for scene in manifest["scenes"]:
                entries = model["scenes"][scene]
                self.assertGreater(entries["checkpoint"]["size_bytes"], 1_000_000)
                self.assertRegex(entries["checkpoint"]["crc32"], r"^[0-9a-f]{8}$")
                self.assertRegex(entries["cameras"]["sha256"], r"^[0-9a-f]{64}$")

    def test_unpinned_archive_cannot_prepare_tier_a_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "dataset.json"
            manifest.write_text(json.dumps({
                "dataset_id": "test", "archive_sha256": None, "archive_format": "zip"
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not pinned"):
                prepare(manifest, root / "data")

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("../escape.txt", "bad")
            with self.assertRaisesRegex(ValueError, "escapes destination"):
                _safe_extract_zip(archive, root / "output")

    def test_zip_path_traversal_after_archive_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "bad-root.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("scene/../escape.txt", "bad")
            with self.assertRaisesRegex(ValueError, "escapes destination"):
                _safe_extract_zip(archive, root / "output", "scene")

    def test_per_scene_gcs_checksums_authorize_and_record_local_sha256(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "scene.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("scene/images/frame.jpg", b"pixels")
            manifest = root / "dataset.json"
            manifest.write_text(json.dumps({
                "schema_version": 2,
                "dataset_id": "test",
                "scenes": {
                    "scene": {
                        "source": _checksum_source(archive),
                        "archive_root": "scene",
                        "image_directory": "images",
                    }
                },
            }), encoding="utf-8")

            inventory = prepare(manifest, root / "data", archive, scene_id="scene")

            self.assertEqual(inventory["scene_id"], "scene")
            self.assertEqual(inventory["source_archive_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertEqual(inventory["source_verification"]["md5_base64"], _checksum_source(archive)["checksums"]["md5_base64"])
            self.assertEqual(inventory["files"], ["images/frame.jpg"])
            self.assertTrue((root / "data" / "raw" / "test" / "scene" / "images" / "frame.jpg").exists())

    def test_per_scene_checksum_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "scene.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("scene/file.txt", "official")
            source = _checksum_source(archive)
            source["checksums"]["crc32c_base64"] = "AAAAAA=="
            manifest = root / "dataset.json"
            manifest.write_text(json.dumps({
                "schema_version": 2,
                "dataset_id": "test",
                "scenes": {"scene": {"source": source, "archive_root": "scene"}},
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "CRC32C mismatch"):
                prepare(manifest, root / "data", archive, scene_id="scene")


if __name__ == "__main__":
    unittest.main()
