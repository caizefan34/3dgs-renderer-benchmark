#!/usr/bin/env python3
"""P16-RB room-pilot gate: which methods may expand to bicycle/garden (prereg section 7).

Criterion = "reproduces a sensible native room result": the run completed, the common
metric pipeline evaluated the full hold-8 test split, quality is finite and far above the
degenerate range, and the run is clean at the common resolution. This is deliberately NOT
the section-10 C0-matched quality box (that box only governs speedup admission against
P12 C0 and is applied later by p16rb_aggregate.py).
"""
import argparse
import json
import math
from pathlib import Path

D = Path("/root/epic05_p16rb")
METHODS = ("FasterGS", "FasterGSFused", "FastGS", "SADGS")
PSNR_SANITY_FLOOR = 25.0  # far below any plausible native room result; catches degenerates only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results, passing = {}, []
    for m in METHODS:
        p = D / "runs" / m / "room" / "P16_RUN.json"
        r = {"record": str(p), "exists": p.exists()}
        if p.exists():
            d = json.loads(p.read_text())
            mc = d.get("metrics_common") or {}
            checks = {
                "returncode_zero": d.get("returncode") == 0,
                "full_test_split": d.get("n_metrics_evaluated") == d.get("n_metrics_expected") and
                                   (d.get("n_metrics_evaluated") or 0) > 0,
                "psnr_finite_above_floor": bool(mc.get("psnr") is not None and
                                                math.isfinite(mc["psnr"]) and mc["psnr"] >= PSNR_SANITY_FLOOR),
                "ssim_lpips_finite": bool(mc.get("ssim") is not None and mc.get("lpips") is not None and
                                          math.isfinite(mc["ssim"]) and math.isfinite(mc["lpips"])),
                "clean": bool(d.get("clean")),
                "direct_comparable_resolution": bool(d.get("direct_comparable_resolution")),
            }
            r["checks"] = checks
            r["metrics_common"] = mc
            r["training_wall_s"] = d.get("training_wall_s")
            r["pass"] = all(checks.values())
            if r["pass"]:
                passing.append(m)
        else:
            r["pass"] = False
        results[m] = r

    out = {
        "gate": "ROOM_PILOT_EXPANSION",
        "criterion": "prereg section 7 'sensible native room result' = completed + full-split "
                     "common metrics + finite quality above a degenerate floor (PSNR >= 25) + "
                     "clean + common-resolution. NOT the section-10 C0-matched quality box.",
        "psnr_sanity_floor": PSNR_SANITY_FLOOR,
        "methods": results,
        "passing": passing,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"passing": passing,
                      "detail": {m: r.get("checks") for m, r in results.items() if r["exists"]}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
