#!/usr/bin/env python3
"""P16-RB: dump the hold-8 expected test-image names for a common_data scene.

Rule: sort the images.bin image names, take positions where pos % 8 == 0.
Writes {"scene":..., "rule":..., "test_step":8, "n_total":..., "test_names":[...]}.
"""
import json
import sys
from pathlib import Path
from struct import unpack

COMMON = Path("/root/epic05_p16rb/common_data/mipnerf360")
TEST_STEP = 8


def main():
    scene = sys.argv[1]
    out = Path(sys.argv[2])
    names = []
    with open(COMMON / scene / "sparse" / "0" / "images.bin", "rb") as f:
        n = unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4 + 32 + 24 + 4)
            nm = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                nm += ch
            names.append(nm.decode())
            npts = unpack("<Q", f.read(8))[0]
            f.read(24 * npts)
    order = sorted(range(len(names)), key=lambda i: names[i])
    test = [names[i] for pos, i in enumerate(order) if pos % TEST_STEP == 0]
    out.write_text(json.dumps({
        "scene": scene, "rule": "hold8_sorted_image_name_idx_mod_8_eq_0",
        "test_step": TEST_STEP, "n_total": len(names),
        "n_test": len(test), "test_names": test,
    }, indent=2), encoding="utf-8")
    print(f"[{scene}] n_total={len(names)} n_test={len(test)} -> {out}")


if __name__ == "__main__":
    main()
