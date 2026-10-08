#!/usr/bin/env python3
"""P16-RB: assemble curves/<method>/<scene>.json from a curve-replay run.

x-axis = training-loop wall time to reach the checkpoint, parsed from the framework's own
progress stream (tqdm `[elapsed<eta]`), NOT the command wall time:
  graphdeco: the native recipe has no inline evaluation, so the tqdm elapsed at iteration k
             is the training time (own sub-second checkpoint saves included, as native).
  nerficg  : our curve protocol adds inline test-renders at each intermediate checkpoint;
             their measured durations (checkpoint-file mtime -> last rendered image mtime)
             are subtracted so both lineages report native training-loop time. The final
             point lands on the same scale as timings.txt `Time:`.

y-axis = the ONE common metric pipeline output for that checkpoint's test render.
"""
import argparse
import json
import re
import time
from pathlib import Path

PROG_RE = re.compile(r"(?:training|Training progress):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<")


def parse_elapsed(s: str) -> int:
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"bad elapsed {s}")


def parse_progress(text: str):
    out = {}
    for m in PROG_RE.finditer(text):
        it, total, el = int(m.group(1)), int(m.group(2)), parse_elapsed(m.group(3))
        out[it] = el
    if not out:
        raise SystemExit("no tqdm progress lines found in train log")
    return out


def elapsed_at(prog: dict, bar_target: int):
    ks = [k for k in prog if k <= bar_target]
    if not ks:
        k = min(prog)
        return prog[k], k
    k = max(ks)
    return prog[k], k


def pause_seconds(outdir: Path, ckpt_iter: int):
    ckpt = outdir / "checkpoints" / f"{ckpt_iter:07d}.pt"
    rgb_dir = outdir / f"test_{ckpt_iter}" / "rgb"
    files = sorted(rgb_dir.glob("*.png"))
    if not ckpt.exists() or not files:
        return None
    t0 = ckpt.stat().st_mtime
    t1 = max(f.stat().st_mtime for f in files)
    return max(0.0, t1 - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--points-json", required=True)
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--materials-dir", required=True)
    ap.add_argument("--nerficg-outdir", default=None,
                    help="needed for nerficg runs: used to measure inline pause durations")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = json.loads(Path(args.points_json).read_text())
    log_text = Path(args.train_log).read_text(errors="replace")
    prog = parse_progress(log_text)
    materials = Path(args.materials_dir)

    # measure inline pauses once (nerficg); the final point's render is post-training and
    # therefore never inside the loop, so only c < native_iterations contribute
    pauses = {}
    if args.nerficg_outdir:
        for q in man["points"]:
            if q["iteration"] < man["native_iterations"]:
                pauses[q["iteration"]] = pause_seconds(Path(args.nerficg_outdir), q["iteration"])
    corrections = [{"iteration": c, "pause_s": v} for c, v in sorted(pauses.items())]

    points = []
    for p in man["points"]:
        it = p["iteration"]
        cumulative_pause = sum(v or 0.0 for c, v in pauses.items() if c < it)
        raw, bar = elapsed_at(prog, it + 1)
        x = raw - cumulative_pause
        met_path = materials / f"metrics_{it}.json"
        met = json.loads(met_path.read_text())
        points.append({
            "fraction": p["fraction"], "iteration": it,
            "training_elapsed_s": round(x, 3),
            "psnr": met["mean"]["psnr"], "ssim": met["mean"]["ssim"], "lpips": met["mean"]["lpips"],
            "elapsed_evidence": {
                "tqdm_bar_iteration": bar, "tqdm_elapsed_s": float(raw),
                "inline_pauses_subtracted_s": round(cumulative_pause, 3),
                "metrics_json": str(met_path), "n_evaluated": met.get("n_evaluated"),
                "metrics_status": met.get("status"),
            },
        })

    deltas = [round(points[i + 1]["training_elapsed_s"] - points[i]["training_elapsed_s"], 3)
              for i in range(len(points) - 1)]
    out = {
        "method": args.method, "scene": args.scene,
        "native_iterations": man["native_iterations"],
        "points": points,
        "protocol": {
            "source": "separate curve replay (fresh run; frozen native recipe; only the "
                      "checkpoint schedule differs from the primary timed run)",
            "x_axis": "training-loop wall time from the framework progress stream; "
                      "nerficg inline curve-render pauses subtracted via file mtimes",
            "train_log": args.train_log, "points_json": args.points_json,
            "materials_dir": str(materials),
            "nerficg_outdir": args.nerficg_outdir,
            "inline_pauses": corrections,
            "x_deltas_between_points_s": deltas,
            "monotonic_x": all(d > 0 for d in deltas),
        },
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"method": args.method, "scene": args.scene,
                      "x_s": [pt["training_elapsed_s"] for pt in points],
                      "psnr": [round(pt["psnr"], 3) for pt in points],
                      "monotonic_x": out["protocol"]["monotonic_x"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
