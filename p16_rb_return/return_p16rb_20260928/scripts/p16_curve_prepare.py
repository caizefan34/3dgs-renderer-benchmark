#!/usr/bin/env python3
"""P16-RB curve-replay preparation: 5-point TTQ plan for one (method, scene).

Point schedule: iteration = round(f * N) + offset for f < 1.0; the final point (f = 1.0) is
always exactly N. The offset exists because nerficg's reset_opacities callback (priority 90,
every OPACITY_RESET_INTERVAL) runs BEFORE the backup save+render callback (priority 1) within
the same loop iteration: a snapshot taken exactly at a reset multiple captures the just-reset
near-transparent state and its test render is spuriously dark (observed PSNR 9.8/10.5 at
6000/12000 vs 31.2 at 18000). Offset -30 puts the snapshot just before each reset multiple,
at full trained quality, without changing the training dynamics.

nerficg : copies the frozen stage config and sets MODEL_NAME. With --offset != 0,
          TRAINING.BACKUP.INTERVAL is pinned to -1 and the real schedule is the
          P16_BACKUP_ITERS env list consumed by the patched copy tree ($D/curve_src/nerficg,
          see curve_schedule.patch); a missing env then fails loudly instead of silently
          falling back to reset-aligned saves. With --offset 0 (legacy), INTERVAL = N/5.
graphdeco: nothing to write; the driver appends --save_iterations <points> to the native
          recipe (train.py always appends the final iteration too). Not affected by the reset
          ordering: their saves precede the in-loop reset by code ordering.

Both: writes the points manifest consumed by run_curves.sh and p16_build_curve.py.
This is a SEPARATE replay run; the primary timed run keeps its frozen config untouched.
"""
import argparse
import json
from pathlib import Path

FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
NERFICG = ("FasterGS", "FasterGSFused")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--native-iterations", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0,
                    help="added to every non-final point iteration; the final point stays exactly N")
    ap.add_argument("--stage-config", default=None, help="nerficg frozen stage config to copy")
    ap.add_argument("--out-config", default=None, help="nerficg curve config destination")
    ap.add_argument("--out-points", required=True)
    args = ap.parse_args()

    N = args.native_iterations
    points = [{"fraction": f, "iteration": N if f >= 1.0 else int(round(f * N)) + args.offset}
              for f in FRACTIONS]
    if len({p["iteration"] for p in points}) != len(points):
        raise SystemExit("non-unique point iterations")

    interval = None
    save_iterations = None
    backup_iters = None
    if args.method in NERFICG:
        if not args.stage_config or not args.out_config:
            raise SystemExit("nerficg requires --stage-config and --out-config")
        import yaml  # only nerficg needs it; graphdeco pair prepare runs under the gsplat env
        cfg = yaml.safe_load(Path(args.stage_config).read_text())
        cfg["TRAINING"]["MODEL_NAME"] = f"p16_curve_{args.method.lower()}_{args.scene}"
        if args.offset:
            cfg["TRAINING"].setdefault("BACKUP", {})["INTERVAL"] = -1
            interval = -1
            backup_iters = [p["iteration"] for p in points if p["fraction"] < 1.0]
        else:
            if N % len(FRACTIONS) != 0:
                raise SystemExit(f"{N} not divisible by {len(FRACTIONS)}")
            interval = N // len(FRACTIONS)
            cfg["TRAINING"].setdefault("BACKUP", {})["INTERVAL"] = interval
        Path(args.out_config).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    else:
        save_iterations = [p["iteration"] for p in points]

    manifest = {
        "method": args.method, "scene": args.scene, "native_iterations": N,
        "fractions": FRACTIONS, "offset": args.offset, "points": points,
        "nerficg_interval": interval, "nerficg_backup_iters_env": backup_iters,
        "graphdeco_save_iterations": save_iterations,
        "rule": "separate curve replay; fresh run with the frozen native recipe; only the "
                "checkpoint schedule (P16_BACKUP_ITERS list / --save_iterations) differs so "
                "the primary timed run stays free of measurement-only I/O; nerficg points sit "
                "offset iterations before the nominal fractions so a snapshot is never taken "
                "in the same iteration as an opacity reset (save priority 1 < reset priority 90)",
    }
    p = Path(args.out_points)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
