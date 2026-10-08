#!/usr/bin/env python3
"""P16-RB probe3: exact per-save callback accounting for a completed curve pair (read-only).

Model (validated on FasterGS room):
  matches per bar: first occurrence = pre-callback elapsed; callback redraws increase elapsed
  at the same bar.  For save it:
     A = max bar <= it      -> e_before = first-match elapsed at A (pre-callback)
     B = max bar <= it + 1  -> e_after  = first-match elapsed at B (post-callback)
     cb_full = e_after - e_before     (training gap A..B negligible: <= 13 iters)
     render_mtime = ckpt end -> last rgb end (only nerficg)
  pure training time x_T(it) = e_after(it) - sum_{c<it} cb_full(c) - cb_full(it)[it<native]
"""
import glob
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

D = Path("/root/epic05_p16rb")
M = sys.argv[1] if len(sys.argv) > 1 else "FasterGS"
SC = sys.argv[2] if len(sys.argv) > 2 else "room"
MAT = D / "curves" / M / (SC + "_materials")

print("==== curve json (current build)")
cpath = D / "curves" / M / (SC + ".json")
if cpath.exists():
    c = json.loads(cpath.read_text())
    for p in c["points"]:
        ev = p["elapsed_evidence"]
        print("  frac", p["fraction"], "it", p["iteration"], "x_now", p["training_elapsed_s"],
              "raw", ev["tqdm_elapsed_s"], "bar", ev["tqdm_bar_iteration"],
              "pauses", ev["inline_pauses_subtracted_s"])

print("==== primary run record")
rp = D / "runs" / M / SC / "P16_RUN.json"
if rp.exists():
    r = json.loads(rp.read_text())
    print("  training_wall_s", r.get("training_wall_s"))
    print("  timing_semantics", r.get("timing_semantics"))
    print("  native_iterations", r.get("native_iterations"))
    print("  evidence", json.dumps(r.get("evidence"))[:500])

print("==== timings.txt candidates")
for pat in [str(D / "curve_src/nerficg/output" / M / ("p16_curve_*_%s_*" % SC) / "timings.txt"),
            str(D / "src/nerficg/output" / M / ("p16_*_%s_*" % SC) / "timings.txt")]:
    for t in glob.glob(pat):
        print("  ", t)
        print("   ", repr(Path(t).read_text(errors="replace")[:400]))

print("==== tqdm accounting")
PROG_RE = re.compile(r"(?:training|Training progress):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<")


def pe(s):
    p = [int(x) for x in s.split(":")]
    return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]


log = (MAT / "train.stdout.log").read_text(errors="replace")
matches = [(int(m.group(1)), pe(m.group(3)), m.start()) for m in PROG_RE.finditer(log)]
first = OrderedDict()
for b, e, pos in matches:
    if b not in first:
        first[b] = e
ks = sorted(first)
print("  matches", len(matches), "unique bars", len(ks), "max bar", ks[-1],
      "elapsed at max", first[ks[-1]], "max elapsed", max(first.values()))

pts = json.loads((MAT / "points.json").read_text())
native = pts["native_iterations"]
outd = None
cands = sorted(glob.glob(str(D / "curve_src/nerficg/output" / M / ("p16_curve_%s_%s_*" % (M.lower(), SC)))))
if cands:
    outd = Path(cands[-1])

cbs = {}
for q in pts["points"]:
    it = q["iteration"]
    A = max(k for k in ks if k <= it)
    Bmax = [k for k in ks if k <= it + 1]
    B = max(Bmax) if Bmax else A
    e_before = first[A]
    if B > A:
        e_after = first[B]
    else:
        e_after = None
        for b, e, pos in matches:
            if b >= A and e > e_before:
                e_after = e
                break
    cb = (e_after - e_before) if e_after is not None else None
    rend = None
    if outd and it < native:
        ck = outd / "checkpoints" / ("%07d.pt" % it)
        rg = sorted((outd / ("test_%d" % it) / "rgb").glob("*.png"))
        if ck.exists() and rg:
            rend = max(f.stat().st_mtime for f in rg) - ck.stat().st_mtime
    cbs[it] = cb
    print("  it", it, "A", A, "e_before", e_before, "B", B, "e_after", e_after,
          "cb_full", cb, "render_mtime", round(rend, 2) if rend else None,
          "nonrender", round(cb - rend, 2) if (cb and rend) else None)

# corrected x
for q in pts["points"]:
    it = q["iteration"]
    cum = sum(cbs[c] for c in cbs if c < it and cbs[c])
    own = cbs[it] if it < native and cbs.get(it) else 0
    xT = (first[max(k for k in ks if k <= it + 1)] if it < native else first[ks[-1]]) - cum - own
    print("  x_T(%d) = %s  (cum_cb_before=%s own=%s)" % (it, xT, cum, own))
