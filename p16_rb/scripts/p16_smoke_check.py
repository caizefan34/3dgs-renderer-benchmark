#!/usr/bin/env python3
"""P16-RB smoke check: brightness + quick PSNR of the dip-fix smoke renders.

usage: python p16_smoke_check.py <curve_src output dir>/FasterGS/p16_smoke_dip_<ts>
Expected: iter 5970 (pre-reset) bright with high quick-PSNR; iter 6000 (just-reset) dark;
iter 6100 final (100 iters after reset) dark-to-recovering.
"""
import sys
import numpy as np
from PIL import Image
from pathlib import Path

outdir = Path(sys.argv[1])
for it in (5970, 6000, 6100):
    d = outdir / f"test_{it}"
    rgbs = sorted((d / "rgb").glob("*.png"))
    if not rgbs:
        print(f"iter {it}: NO RENDERS at {d}")
        continue
    psnrs, gains, paired = [], [], 0
    for f in rgbs:
        a = np.asarray(Image.open(f), dtype=np.float64) / 255.0
        gains.append(float(a.mean()))
        g = d / "rgb_gt" / f.name
        if g.exists():
            b = np.asarray(Image.open(g), dtype=np.float64) / 255.0
            mse = ((a - b) ** 2).mean()
            psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))
            paired += 1
    line = f"iter {it}: renders={len(rgbs)} mean_rgb={np.mean(gains):.4f}"
    if paired:
        line += f" quick_psnr={np.mean(psnrs):.2f} (paired {paired}/{len(rgbs)} by name)"
    else:
        line += " quick_psnr=n/a (name pairing found 0)"
    print(line)
