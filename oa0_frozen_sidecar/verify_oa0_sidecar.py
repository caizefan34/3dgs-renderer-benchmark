#!/usr/bin/env python3
"""Replay the OA0 frozen sidecar intake verification (see VERIFICATION_RECORD.md).

Checks:
  A. 8/8 manifest SHA256 on the archived files
  B. smoke reproduction of UNIT_TEST.json
  C. 384 frozen labels reproduce from N1D_ACTION_RESULTS.parquet
  D. N1D_LABEL_SUMMARY.json reproduces from N1D_LABELS_FROZEN.csv
"""
import hashlib, json, math, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N1D = ROOT / "N1D_RETURN" / "artifacts"

fails = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)

def approx(a, b, rel=1e-9, atol=1e-15):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=atol)

# ---- A. manifest hashes ----
man = json.loads((HERE / "MANIFEST.json").read_text())
bad = [n for n, m in man.items()
       if hashlib.sha256((HERE / n).read_bytes()).hexdigest() != m["sha256"]]
check("A manifest hashes (8 files)", not bad, ",".join(bad))

# ---- B. smoke re-run ----
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "repro.json"
    r = subprocess.run([sys.executable, str(HERE / "oa0_analyze_frozen.py"),
                        "--features", str(HERE / "_synthetic_features.csv"),
                        "--labels", str(HERE / "N1D_LABELS_FROZEN.csv"),
                        "--out", str(out)],
                       capture_output=True, text=True, check=True, cwd=HERE)
    repro = json.loads(out.read_text())
smoke = json.loads(json.loads((HERE / "UNIT_TEST.json").read_text())["stdout"])
repro_prev, smoke_prev = repro["damping_rescue_prevalence"], smoke["prevalence"]
ok = (repro["verdict"] == smoke["verdict"]
      and all(approx(repro_prev[s], smoke_prev[s]) for s in ("room", "bicycle"))
      and all(approx(repro["transfer"][k][m], smoke["transfer"][k][m])
              for k in ("room_to_bicycle", "bicycle_to_room") for m in ("roc_auc", "average_precision", "brier")))
check("B smoke reproduces UNIT_TEST.json", ok, repro["verdict"])

# ---- C. label reconstruction ----
lab = pd.read_csv(HERE / "N1D_LABELS_FROZEN.csv")
df = pd.read_parquet(N1D / "N1D_ACTION_RESULTS.parquet")
fam = df[df.action.isin(["A0", "A1-1.00", "A1-0.75", "A1-0.50", "A1-0.25"])].copy()
fam["a"] = fam.action.map({"A0": 0.0, "A1-1.00": 1.0, "A1-0.75": 0.75, "A1-0.50": 0.50, "A1-0.25": 0.25})
fam["progress"] = fam.delta_fused
ranked = fam.sort_values("progress", ascending=False)

def extreme(alphas, name):
    a = ranked[ranked.a.isin(alphas)].groupby(["scene", "checkpoint", "anchor_id"], as_index=False).first()
    return a[["scene", "checkpoint", "anchor_id", "a", "progress"]].rename(
        columns={"a": f"{name}_alpha", "progress": f"{name}_p"})

all5, damped = extreme([0.0, 0.25, 0.5, 0.75, 1.0], "all"), extreme([0.0, 0.25, 0.5, 0.75], "damped")
native = fam[fam.action == "A1-1.00"][["scene", "checkpoint", "anchor_id", "progress"]].rename(columns={"progress": "native_p"})
m = (lab.merge(native, on=["scene", "checkpoint", "anchor_id"], validate="one_to_one")
        .merge(all5, on=["scene", "checkpoint", "anchor_id"], validate="one_to_one")
        .merge(damped, on=["scene", "checkpoint", "anchor_id"], validate="one_to_one"))
eq = lambda a, b: bool(np.isclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=1e-15).all())
checks = {
    "native_progress": eq(m.native_progress, m.native_p),
    "damped_best_progress": eq(m.damped_best_progress, m.damped_p),
    "overshoot_gain": eq(m.overshoot_gain, m.damped_p - m.native_p),
    "best_progress_all_alpha": eq(m.best_progress_all_alpha, m.all_p),
    "best_alpha": eq(m.best_alpha, m.all_alpha),
    "native_harmful": bool((m.native_harmful.astype(bool) == (m.native_progress < 0)).all()),
    "damping_rescue": bool((m.damping_rescue.astype(bool) == (m.all_p > m.native_p)).all()),
}
check("C labels reproduce from N1D parquet (384 rows)",
      len(m) == 384 and all(checks.values()), ",".join(k for k, v in checks.items() if not v))

# ---- D. summary reproduction ----
def flat(x):
    return {"n": int(len(x)),
            "native_harmful_count": int(x.native_harmful.sum()),
            "native_harmful_rate": float(x.native_harmful.mean()),
            "damping_rescue_count": int(x.damping_rescue.sum()),
            "damping_rescue_rate": float(x.damping_rescue.mean()),
            "median_overshoot_gain": float(x.overshoot_gain.median()),
            "mean_overshoot_gain": float(x.overshoot_gain.mean()),
            "best_alpha_counts": {str(k): int(v) for k, v in x.best_alpha.value_counts().sort_index().items()}}

frozen_sum = json.loads((HERE / "N1D_LABEL_SUMMARY.json").read_text())
okd = True
for scene in ("room", "bicycle"):
    x = lab[lab.scene == scene]
    f, got = frozen_sum[scene], flat(x)
    for k, v in f.items():
        if k == "by_checkpoint":
            continue
        okd &= approx(v, got[k]) if isinstance(v, (int, float)) else v == got[k]
    for ck, blk in f["by_checkpoint"].items():
        g = flat(x[x.checkpoint == int(ck)])
        for k, v in blk.items():
            okd &= approx(v, g[k]) if isinstance(v, (int, float)) else v == g[k]
check("D summary reproduces from frozen labels", okd)

print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED: {fails}")
sys.exit(1 if fails else 0)
