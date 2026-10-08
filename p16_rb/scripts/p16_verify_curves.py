#!/usr/bin/env python3
"""P16-RB: verify the 12 rebuilt curve JSONs. Exit 1 if any check fails.

Checks per pair: 5 points, strictly monotonic x, |x_final - primary training_wall| <= 5%;
nerficg pairs must have the measured callback-subtraction total; graphdeco pairs must have
the alpha-calibration block (alpha = native CUDA-event total / wall-after-saves).
"""
import json
import sys
from pathlib import Path

R = Path("/root/epic05_p16rb")
METHODS = ["FasterGS", "FasterGSFused", "FastGS", "SADGS"]
SCENES = ["room", "bicycle", "garden"]
NERFICG = {"FasterGS", "FasterGSFused"}
bad = 0
for m in METHODS:
    for s in SCENES:
        cp = R / "curves" / m / f"{s}.json"
        pp = R / "runs" / m / s / "P16_RUN.json"
        if not cp.exists() or not pp.exists():
            print("MISSING", m, s, cp.exists(), pp.exists())
            bad += 1
            continue
        c = json.loads(cp.read_text())
        r = json.loads(pp.read_text())
        pts = c["points"]
        xf = pts[-1]["training_elapsed_s"]
        wall = r["training_wall_s"]
        drift = (xf - wall) / wall
        mono = c["protocol"]["monotonic_x"]
        sub = c["protocol"].get("callbacks_subtracted_total_s")
        gc = c["protocol"].get("graphdeco_calibration") or {}
        alpha = gc.get("alpha")
        ev = gc.get("event_total_s")
        ok = mono and len(pts) == 5 and abs(drift) <= 0.05
        if m in NERFICG and sub is None:
            ok = False
        if m not in NERFICG and (alpha is None or ev is None or not (0.9 < alpha < 1.1)):
            ok = False
        if not ok:
            bad += 1
        print(("OK  " if ok else "BAD ") + f"{m}/{s}: npts={len(pts)} mono={mono} "
              f"x_final={xf} wall={round(wall, 2)} drift={drift * 100:+.2f}% cb_total={sub} "
              f"alpha={alpha}")
print("VERIFY_CURVES_BAD", bad)
sys.exit(1 if bad else 0)
