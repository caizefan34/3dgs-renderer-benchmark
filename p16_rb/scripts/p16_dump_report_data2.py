#!/usr/bin/env python3
"""Second read-only dump: census final medians + curve point structure."""
import json
import os

D = "/root/epic05_p16rb"
A = os.path.join(D, "artifacts")
METHODS = ["FasterGS", "FasterGSFused", "FastGS", "SADGS"]
SCENES = ["room", "bicycle", "garden"]

c = json.load(open(os.path.join(A, "P16_BOTTLENECK_CENSUS.json"), encoding="utf-8"))
print("=== CENSUS FINAL MEDIANS ===")
for r in c["rows"]:
    f = r["profiles"]["final"]["median_ms"]
    mi = r["profiles"]["mid"]["median_ms"]
    print("%s/%s it=%s/%s exec=%.4f dens=%.4f midtot=%.4f fintot=%.4f | fin fwd=%.4f bwd=%.4f opt=%.4f bof=%.4f proj=%.4f sort=%.4f loss=%.4f other=%.4f densc=%.4f" % (
        r["method"], r["scene"], r["native_iterations"], r["p12_iterations"],
        r["exec_share_final"], r["density_share_final"],
        mi["total"], f["total"],
        f["raster_forward"], f["raster_backward"], f["optimizer"],
        f["backward_optimizer_fused"], f["projection_preprocess"],
        f["intersection_sort"], f["loss"], f["other"], f["density_control"]))

print()
print("=== CURVE POINTS ===")
for m in METHODS:
    for s in SCENES:
        cu = json.load(open(os.path.join(D, "curves", m, s + ".json"), encoding="utf-8"))
        pts = cu.get("points") or []
        print("--- %s/%s  top keys: %s" % (m, s, list(cu.keys())))
        if pts:
            print("    point keys:", list(pts[0].keys()))
        for pt in pts:
            print("   ", json.dumps(pt)[:300])
