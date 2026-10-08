#!/usr/bin/env python3
"""Assemble artifacts/P16_DATA_HEAL_garden.json from the heal + rerun evidence.

Run after garden_heal_rerun.sh logs GARDEN_HEAL_RERUN_DONE.
"""
import hashlib
import json
import sys
from pathlib import Path

D = Path("/root/epic05_p16rb")
ART = D / "artifacts"
OUT = ART / "P16_DATA_HEAL_garden.json"
METHODS = ["FasterGS", "FasterGSFused", "FastGS", "SADGS", "P12_C0"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    man = load(D / "garden_repair/official/MANIFEST_garden_official.json")
    common_rep = load(D / "common_data/mipnerf360/P16_COMMON_DATA_BUILD.json")
    pre_backup = (D / "backups/garden_corrupt_20260928/SHA256SUMS.txt").read_text().splitlines()

    old, new = {}, {}
    for m in METHODS:
        r_old = load(D / "runs" / m / "garden_flatdata_corrupt/P16_RUN.json")
        r_new = load(D / "runs" / m / "garden/P16_RUN.json")
        if r_old:
            old[m] = {"metrics_common": r_old.get("metrics_common"),
                      "training_wall_s": r_old.get("training_wall_s"),
                      "clean": r_old.get("clean")}
        if r_new:
            new[m] = {"metrics_common": r_new.get("metrics_common"),
                      "training_wall_s": r_new.get("training_wall_s"),
                      "command_wall_s": r_new.get("command_wall_s"),
                      "peak_vram_mb": r_new.get("peak_vram_mb"),
                      "num_gaussians_final": r_new.get("num_gaussians_final"),
                      "clean": r_new.get("clean")}

    rec = {
        "artifact": "P16_DATA_HEAL_garden",
        "date": "2026-09-28",
        "decision": "operator-approved heal + full garden rerun (all 4 external methods + P12_C0)",
        "corruption": {
            "source_of_truth": "shared dataset /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/garden/sparse/0",
            "onset": "2026-09-16T20:24-20:32 rebuild; garden-only (room/bicycle/flowers/stump/treehill healthy)",
            "symptom": "points3D.bin 1,839,236-point flat cloud (z std ~0); images.bin stripped of 2D "
                       "keypoints (15,733B vs official 47,011,789B); points3D.ply same 27B/pt flat format "
                       "(49,659,607B); all garden runs on the box since 9/16 trained from it",
            "consequence": "flat-data garden records for all methods retained verbatim under "
                           "runs/<method>/garden_flatdata_corrupt/ (metrics reproduced below)",
            "pre_heal_backups": {
                "dir": str(D / "backups/garden_corrupt_20260928"),
                "sha256sums": pre_backup,
            },
        },
        "official_source": {
            "url": "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
            "fetch_script": str(D / "scripts/fetch_garden_sparse.py"),
            "members": man["members"] if man else None,
            "verification": [
                "CRC32 + size checked at fetch against the zip central directory",
                "185/185 image names match the repo tree exactly; poses match "
                "data/official/mipnerf360/garden/cameras.json (maxdiff ~1e-16)",
                "sample image DSC07956.JPG byte-identical to the repo copy "
                "(sha256 4b0d24079aa8c09989cfaff67bcb1d1bd3dc4887c2b768403920ea7f6cb9f6e0)",
            ],
        },
        "heal_steps": {
            "log": str(D / "logs/garden_heal.log"),
            "H1_backup": "corrupt repo+common sparse files copied to backups/garden_corrupt_20260928",
            "H2_install": "official cameras.bin / images.bin / points3D.bin installed into repo sparse/0",
            "H3_ply_regen": {
                "method": "scene.dataset_readers.read_points3D_binary + storePly (FastGS loader code path)",
                "generator_validation": "bicycle regen BYTE-IDENTICAL to shipped bicycle points3D.ply",
                "garden_ply_bytes": (D / "common_data/mipnerf360/garden/sparse/0/points3D.ply").stat().st_size,
                "garden_ply_sha256": sha256_file(D / "common_data/mipnerf360/garden/sparse/0/points3D.ply"),
            },
            "H4_pycolmap": "healed repo sparse loads strictly: 185 images, 138766 points3D, PINHOLE 5187x3361",
            "H5_restage": "garden-only p16_build_common_data.py run (argv scene filter added; report merged)",
            "H6_verify": {
                "repo_sparse_sha256": {f: sha256_file(D / "garden_repair/official/sparse/0" / f)
                                       for f in ("cameras.bin", "images.bin", "points3D.bin")},
                "common_sparse_sha256": {f: sha256_file(D / "common_data/mipnerf360/garden/sparse/0" / f)
                                         for f in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply")},
                "build_report_garden": (common_rep or {}).get("garden"),
            },
        },
        "rerun": {
            "driver": str(D / "scripts/garden_heal_rerun.sh"),
            "log": str(D / "logs/garden_heal_rerun.log"),
            "protocol": "flat-data evidence archived (renames); external methods rerun with the exact "
                        "stage-2 recipe on GPU 2 (wrapper + vram sampler + captures + finalize on GPU 3); "
                        "P12_C0 rerun with the exact archived recipe + common-split render + canonical "
                        "metrics + record rebuilt via p16_build_p12_record.py",
            "flat_data_records": old,
            "healed_records": new,
        },
        "verdict_note": "cross-method garden comparison must use the healed records; flat-data records "
                        "are corruption evidence, not results",
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("WROTE", OUT)
    for m in METHODS:
        o = (old.get(m) or {}).get("metrics_common")
        n = (new.get(m) or {}).get("metrics_common")
        print(f"{m:14s} old {o} -> new {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
