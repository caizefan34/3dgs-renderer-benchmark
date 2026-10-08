#!/usr/bin/env python3
"""P16-RB: assemble curves/<method>/<scene>.json from a curve-replay run.

x-axis = training-loop wall time to reach the checkpoint, parsed from the framework's own
progress stream (tqdm `[elapsed<eta]`), NOT the command wall time, and aligned with the
primary wall semantics of each lineage:

nerficg curve protocol adds a per-iteration inline callback (checkpoint save + test render +
metric evaluation) at each intermediate point; its full cost is measured and subtracted so
the axis matches the primary runs (which contain no intermediate callbacks; primary
semantics: `Time:` = training-callback wall, final render/checkpoint excluded).

graphdeco primary semantics are CUDA-event based (`train.py Training time:` = forward+loss+
backward + densification/pruning/optimizer, excluding rendering/checkpoint serialization and
per-iteration CPU overhead). The curve loop-wall x is therefore (a) reduced by the measured
protocol save cost and (b) scaled by alpha = event_total / wall_remaining so the endpoint
equals the run's own event total (uniform per-iteration overhead assumption; raw values and
alpha recorded as evidence).

Measurement details (validated 2026-09-28 against timings.txt `_save_intermediate_checkpoint`
totals for nerficg, agreement <= 1.7%, and against `Training time:` for graphdeco):
  - tqdm redraws during an inline callback re-print the SAME bar with updated wall-clock
    elapsed, so the FIRST occurrence of a bar value is the pre-callback elapsed.
  - per save it: A = last bar refresh <= it (pre-callback), B = first refresh beyond it
    (post-callback); callback cost = first[B] - first[A], integer-second precision.
  - x(it) = [first[A](it) - sum of save costs of c < it] * alpha.

y-axis = the ONE common metric pipeline output for that checkpoint's test render.
"""
import argparse
import json
import re
import time
from pathlib import Path

PROG_RE = re.compile(r"(?:training|Training progress):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<")
TIMING_RE = re.compile(r"_save_intermediate_checkpoint:\s*\n\s*Total execution time: (\d+):(\d+):(\d+)")
EVENT_RE = re.compile(r"Training time:\s*([0-9.eE+-]+)")


def parse_elapsed(s: str) -> int:
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"bad elapsed {s}")


def parse_progress(text: str):
    """first = earliest match per bar (pre-callback), last = latest match per bar."""
    first, last = {}, {}
    for m in PROG_RE.finditer(text):
        it, el = int(m.group(1)), parse_elapsed(m.group(3))
        if it not in first:
            first[it] = el
        last[it] = el
    if not first:
        raise SystemExit("no tqdm progress lines found in train log")
    return first, last


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
                    help="nerficg runs: inline-callback render pauses measured for evidence")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = json.loads(Path(args.points_json).read_text())
    log_text = Path(args.train_log).read_text(errors="replace")
    first, _last = parse_progress(log_text)
    ks = sorted(first)
    native = man["native_iterations"]
    materials = Path(args.materials_dir)
    nerficg = args.nerficg_outdir is not None

    def bar_at_or_before(target):
        cands = [k for k in ks if k <= target]
        return max(cands) if cands else min(ks)

    def callback_cost(it):
        a = bar_at_or_before(it)
        e0 = first[a]
        bs = [k for k in ks if k <= it + 1]
        b = max(bs) if bs else a
        if b > a:
            return first[b] - e0
        for k in ks:
            if k >= a and first[k] > e0:
                return first[k] - e0
        return None

    # protocol-added intermediate saves: fully excluded from the native semantics of both
    # lineages (primary timed runs contain no intermediate checkpoints)
    saves = [q["iteration"] for q in man["points"] if q["iteration"] < native]
    cb = {}
    for c in saves:
        v = callback_cost(c)
        if v is not None:
            cb[c] = v

    corrections = []
    for c in saves:
        if c not in cb:
            continue
        rend = None
        if nerficg:
            rend = pause_seconds(Path(args.nerficg_outdir), c)
        corrections.append({
            "iteration": c, "callback_full_s": cb[c],
            "render_mtime_s": round(rend, 2) if rend is not None else None,
            "callback_nonrender_s": round(cb[c] - rend, 2) if rend is not None else None,
        })

    event_s = None
    if not nerficg:
        em = EVENT_RE.search(log_text)
        if em:
            event_s = float(em.group(1))

    collected = []
    for p in man["points"]:
        it = p["iteration"]
        a = bar_at_or_before(it)
        raw = float(first[a])
        cum = sum(cb[c] for c in saves if c < it)
        met_path = materials / f"metrics_{it}.json"
        met = json.loads(met_path.read_text())
        collected.append({
            "fraction": p["fraction"], "iteration": it,
            "base_s": raw - cum, "raw": raw, "cum": cum, "bar": a,
            "met": met, "met_path": met_path,
        })

    alpha = 1.0
    if event_s is not None:
        denom = collected[-1]["base_s"]
        if denom > 0:
            alpha = event_s / denom

    points = []
    for c in collected:
        x = c["base_s"] * alpha
        ev = {
            "tqdm_bar_iteration": c["bar"], "tqdm_elapsed_s": c["raw"],
            "saves_subtracted_s": round(c["cum"], 3),
            "subtraction": ("protocol inline callbacks (save+render+eval) subtracted; axis aligned "
                            "with nerficg timings.txt Time: semantics" if nerficg else
                            "protocol saves subtracted; wall scaled by alpha to match native "
                            "CUDA-event Training time: semantics"),
            "metrics_json": str(c["met_path"]), "n_evaluated": c["met"].get("n_evaluated"),
            "metrics_status": c["met"].get("status"),
        }
        if c["iteration"] in cb:
            ev["own_callback_s"] = cb[c["iteration"]]
            if nerficg:
                rend = pause_seconds(Path(args.nerficg_outdir), c["iteration"])
                ev["own_render_mtime_s"] = round(rend, 2) if rend is not None else None
        if alpha != 1.0:
            ev["x_before_alpha_s"] = round(c["base_s"], 3)
        points.append({
            "fraction": c["fraction"], "iteration": c["iteration"],
            "training_elapsed_s": round(x, 3),
            "psnr": c["met"]["mean"]["psnr"], "ssim": c["met"]["mean"]["ssim"],
            "lpips": c["met"]["mean"]["lpips"],
            "elapsed_evidence": ev,
        })

    deltas = [round(points[i + 1]["training_elapsed_s"] - points[i]["training_elapsed_s"], 3)
              for i in range(len(points) - 1)]
    protocol = {
        "source": "separate curve replay (fresh run; frozen native recipe; only the "
                  "checkpoint schedule differs from the primary timed run)",
        "x_axis": ("training-loop wall from the framework progress stream; nerficg protocol "
                   "inline callbacks (save+render+eval) measured via first-match tqdm redraws "
                   "and subtracted" if nerficg else
                   "training-loop wall from the framework progress stream; protocol saves "
                   "subtracted and axis scaled by alpha to the run's own CUDA-event "
                   "Training time: total"),
        "train_log": args.train_log, "points_json": args.points_json,
        "materials_dir": str(materials),
        "nerficg_outdir": args.nerficg_outdir,
        "callbacks_measured": corrections,
        "x_deltas_between_points_s": deltas,
        "monotonic_x": all(d > 0 for d in deltas),
    }
    if nerficg:
        tsum = sum(cb[c] for c in saves if c in cb)
        protocol["callbacks_subtracted_total_s"] = tsum
        tp = Path(args.nerficg_outdir) / "timings.txt"
        if tp.exists():
            m = TIMING_RE.search(tp.read_text(errors="replace"))
            if m:
                tt = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                protocol["timings_txt_save_intermediate_total_s"] = tt
                protocol["timings_vs_tqdm_delta_s"] = round(tt - tsum, 1)
    else:
        protocol["saves_subtracted_total_s"] = sum(cb[c] for c in saves if c in cb)
        protocol["graphdeco_calibration"] = {
            "event_total_s": event_s,
            "alpha": round(alpha, 4),
            "wall_after_saves_s": round(collected[-1]["base_s"], 3),
            "basis": "alpha = native CUDA-event Training time: / (loop wall - protocol saves); "
                     "uniform per-iteration overhead assumption",
        }
        if event_s is None:
            protocol["graphdeco_calibration"]["warning"] = "Training time: not found; alpha=1"

    out = {
        "method": args.method, "scene": args.scene,
        "native_iterations": native,
        "points": points,
        "protocol": protocol,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"method": args.method, "scene": args.scene,
                      "x_s": [pt["training_elapsed_s"] for pt in points],
                      "psnr": [round(pt["psnr"], 3) for pt in points],
                      "alpha": round(alpha, 4),
                      "monotonic_x": protocol["monotonic_x"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
