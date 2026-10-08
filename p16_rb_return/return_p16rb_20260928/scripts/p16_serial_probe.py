#!/usr/bin/env python3
"""P16-RB probe: quantify checkpoint serialization time included in tqdm raw elapsed.

For each nerficg curve save point it:
  bar_before = last tqdm bar <= it (pre-pause), bar_after = last tqdm bar <= it+1
  delta_elapsed = elapsed(bar_after) - elapsed(bar_before)
                = T(bar_before..bar_after) + serialization + inline_render
  inline_render measured via mtimes (ckpt end -> last rgb end)
  => serialization ~= delta_elapsed - inline_render - tiny_train_gap

Also dumps curve json evidence + primary run record + timings.txt for cross-check.
Read-only.
"""
import glob
import json
import re
import sys
import time
from pathlib import Path

D = Path("/root/epic05_p16rb")
M = sys.argv[1] if len(sys.argv) > 1 else "FasterGS"
SC = sys.argv[2] if len(sys.argv) > 2 else "room"

PROG_RE = re.compile(r"(?:training|Training progress):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<")


def pe(s):
    p = [int(x) for x in s.split(":")]
    return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]


MAT = D / "curves" / M / (SC + "_materials")
log_text = (MAT / "train.stdout.log").read_text(errors="replace")
prog = {}
for m in PROG_RE.finditer(log_text):
    prog[int(m.group(1))] = pe(m.group(3))
ks = sorted(prog)
print("PROG lines:", len(ks), "min", ks[0], "max", ks[-1], "elapsed_max", prog[ks[-1]])
gaps = [ks[i + 1] - ks[i] for i in range(len(ks) - 1)]
if gaps:
    sg = sorted(gaps)
    print("bar-gap: max", max(gaps), "median", sg[len(sg) // 2], "n>50", sum(1 for g in gaps if g > 50))

pts = json.loads((MAT / "points.json").read_text())
print("native_iterations", pts.get("native_iterations"))
print("points", [(p["fraction"], p["iteration"]) for p in pts["points"]])

outd = None
if M in ("FasterGS", "FasterGSFused"):
    cands = sorted(glob.glob(str(D / "curve_src/nerficg/output" / M / ("p16_curve_%s_%s_*" % (M.lower(), SC)))))
    outd = Path(cands[-1]) if cands else None
print("outdir", outd)

ser = {}
for q in pts["points"]:
    it = q["iteration"]
    print("=== save point", it)
    if outd:
        ck = outd / "checkpoints" / ("%07d.pt" % it)
        rgbs = sorted((outd / ("test_%d" % it) / "rgb").glob("*.png"))
        if ck.exists() and rgbs:
            t0 = ck.stat().st_mtime
            t1 = max(f.stat().st_mtime for f in rgbs)
            print("   ckpt_mtime", time.strftime("%H:%M:%S", time.localtime(t0)),
                  "rgb_last", time.strftime("%H:%M:%S", time.localtime(t1)),
                  "pause_s", round(t1 - t0, 2), "n_rgb", len(rgbs))
        else:
            print("   ckpt/rgb missing", ck.exists(), len(rgbs))
    near = [(k, prog[k]) for k in ks if it - 40 <= k <= it + 40]
    print("   tqdm near:", near)
    below = [k for k in ks if k <= it]
    after = [k for k in ks if k <= it + 1]
    kb = max(below) if below else None
    ka = max(after) if after else None
    if kb is not None and ka is not None and outd:
        t0 = (outd / "checkpoints" / ("%07d.pt" % it)).stat().st_mtime
        rgbs = sorted((outd / ("test_%d" % it) / "rgb").glob("*.png"))
        t1 = max(f.stat().st_mtime for f in rgbs)
        pause = t1 - t0
        d_el = prog[ka] - prog[kb]
        s = d_el - pause
        ser[it] = s
        print("   bar_before", kb, prog[kb], "bar_after", ka, prog[ka],
              "delta_elapsed", d_el, "minus_rgb_pause", round(pause, 2),
              "=> ser_est_s", round(s, 2), "(train_gap <= %.2fs)" % ((ka - kb) / 50.0))

cpath = D / "curves" / M / (SC + ".json")
if cpath.exists():
    c = json.loads(cpath.read_text())
    print("CURVE JSON", cpath)
    for p in c["points"]:
        ev = p["elapsed_evidence"]
        print("   frac", p["fraction"], "it", p["iteration"], "x", p["training_elapsed_s"],
              "bar", ev["tqdm_bar_iteration"], "raw", ev["tqdm_elapsed_s"],
              "cum_pause", ev["inline_pauses_subtracted_s"])
    print("   x_deltas", c["protocol"]["x_deltas_between_points_s"])

for t in glob.glob(str(D / "curve_src/nerficg/output" / M / ("p16_curve_*_%s_*" % SC) / "timings.txt")):
    print("CURVE TIMINGS", t)
    print(Path(t).read_text(errors="replace")[:600])

rp = D / "runs" / M / SC / "P16_RUN.json"
if rp.exists():
    r = json.loads(rp.read_text())
    print("PRIMARY P16_RUN", rp)
    for k in ("method", "scene", "training_wall_s", "timing_semantics", "native_iterations",
              "num_gaussians_final", "metrics_common", "clean", "direct_comparable_resolution"):
        if k in r:
            print("   ", k, "=", r[k])
    if "evidence" in r:
        print("    evidence:", json.dumps(r["evidence"])[:400])

print("SUM ser_est over saves:", round(sum(ser.values()), 2), ser)
