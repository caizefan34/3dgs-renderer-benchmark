#!/usr/bin/env python3
"""P16-RB: build the common-resolution (P12 1080p protocol) datasets for external methods.

For each scene:
  input  : /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/<scene>/{images,sparse/0}
  output : /root/epic05_p16rb/common_data/mipnerf360/<scene>/{images,sparse/0}
    - images  : every original image PIL-LANCZOS resized to exactly 1920x1080 (same pixel
                content as P12 GTDataset GT), written as lossless PNG with .png names
    - sparse/0: cameras.bin rewritten with per-axis scaled intrinsics (PINHOLE):
                  fx' = fx * 1920/W0, fy' = fy * 1080/H0, cx' = 960.0, cy' = 540.0
                images.bin copied with image names rewritten .JPG -> .png (extrinsics unchanged)
                points3D.bin / points3D.ply copied byte-identical (init point cloud unchanged)
This reproduces the P12 camera model exactly: centered principal point, fov derived from
the original focal + original image size (resize_cameras semantics), 1920x1080 output.
"""
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

from PIL import Image

SRC = Path("/root/3dgs-renderer-benchmark/data/datasets/mipnerf360")
DST = Path("/root/epic05_p16rb/common_data/mipnerf360")
SCENES = ("room", "bicycle", "garden")
TARGET_W, TARGET_H = 1920, 1080
PINHOLE_ID = 1


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_cameras_bin(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        cams = []
        for _ in range(n):
            cid, mid = struct.unpack("<ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            nparam = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}[mid]
            params = struct.unpack("<" + "d" * nparam, f.read(8 * nparam))
            cams.append({"id": cid, "model": mid, "w": w, "h": h, "params": params})
    return cams


def write_cameras_bin(path: Path, cams):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for c in cams:
            f.write(struct.pack("<ii", c["id"], PINHOLE_ID))
            f.write(struct.pack("<QQ", c["w"], c["h"]))
            f.write(struct.pack("<4d", *c["params"]))


def rewrite_images_bin(src: Path, dst: Path):
    """Copy images.bin, rewriting image_name extension .JPG/.jpg -> .png. Returns (n, names)."""
    names = []
    with open(src, "rb") as f, open(dst, "wb") as g:
        n = struct.unpack("<Q", f.read(8))[0]
        g.write(struct.pack("<Q", n))
        for _ in range(n):
            head = f.read(4 + 32 + 24 + 4)
            g.write(head)
            nm = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                nm += ch
            name = nm.decode()
            new = str(Path(name).with_suffix(".png"))
            names.append(new)
            g.write(new.encode() + b"\x00")
            npts = struct.unpack("<Q", f.read(8))[0]
            g.write(struct.pack("<Q", npts))
            g.write(f.read(24 * npts))
    return n, names


def main():
    report = {}
    for scene in (sys.argv[1:] or list(SCENES)):
        assert scene in SCENES, f"unknown scene {scene!r}"
        s_dir = SRC / scene
        d_dir = DST / scene
        (d_dir / "images").mkdir(parents=True, exist_ok=True)
        (d_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)

        cams = read_cameras_bin(s_dir / "sparse" / "0" / "cameras.bin")
        assert len(cams) == 1 and cams[0]["model"] == 1, f"{scene}: expected one PINHOLE camera"
        c = cams[0]
        fx, fy, cx, cy = c["params"]
        W0, H0 = c["w"], c["h"]
        sx, sy = TARGET_W / W0, TARGET_H / H0
        fx2, fy2 = fx * sx, fy * sy
        assert abs(cx - W0 / 2) < 1e-6 and abs(cy - H0 / 2) < 1e-6, f"{scene}: pp not centered"
        assert abs(fx2 * (2 * (W0 / (2 * fx))) - TARGET_W) < 1e-3  # fov identity check
        new_cams = [{"id": c["id"], "w": TARGET_W, "h": TARGET_H, "params": (fx2, fy2, TARGET_W / 2, TARGET_H / 2)}]
        write_cameras_bin(d_dir / "sparse" / "0" / "cameras.bin", new_cams)

        n_img, new_names = rewrite_images_bin(s_dir / "sparse" / "0" / "images.bin",
                                             d_dir / "sparse" / "0" / "images.bin")
        for extra in ("points3D.bin", "points3D.ply"):
            p = s_dir / "sparse" / "0" / extra
            if p.exists():
                shutil.copy2(p, d_dir / "sparse" / "0" / extra)

        src_imgs = sorted(p for p in (s_dir / "images").iterdir() if p.is_file())
        written = []
        for p in src_imgs:
            with Image.open(p) as im:
                out = im.resize((TARGET_W, TARGET_H), Image.LANCZOS)
                op = d_dir / "images" / (p.stem + ".png")
                out.save(op, format="PNG", compress_level=6)
                written.append(op.name)
        missing = sorted(set(new_names) - set(written))
        report[scene] = {
            "n_colmap_images": n_img,
            "n_images_written": len(written),
            "names_missing": missing[:5],
            "source_size": [W0, H0],
            "focal_in": [fx, fy],
            "focal_out_1920x1080": [fx2, fy2],
            "scale_xy": [sx, sy],
            "cameras_bin_sha256": sha256_file(d_dir / "sparse" / "0" / "cameras.bin"),
            "images_bin_sha256": sha256_file(d_dir / "sparse" / "0" / "images.bin"),
            "points3D_bin_sha256": sha256_file(d_dir / "sparse" / "0" / "points3D.bin"),
            "first_image_sha256": sha256_file(d_dir / "images" / written[0]),
        }
        print(f"[{scene}] {len(written)} images -> {TARGET_W}x{TARGET_H}; fx {fx:.3f}->{fx2:.3f} fy {fy:.3f}->{fy2:.3f}",
              flush=True)
    out = DST / "P16_COMMON_DATA_BUILD.json"
    merged = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    merged.update(report)
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("BUILD_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
