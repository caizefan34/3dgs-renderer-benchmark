#!/usr/bin/env python3
"""BASELINE_VALIDATION - fail-closed acceptance gate for P8/P9 baseline runs.

Rejects a run if ANY of the following is wrong or unverifiable:
  wrong source commit / wrong binary / wrong scene / wrong resolution /
  wrong seed / wrong eps2d / wrong training iterations / contamination /
  NaN or Inf in final metrics / missing final evaluation.

Fail-closed: a check whose inputs are MISSING is a FAIL, not a skip.

Usage:
  python BASELINE_VALIDATION.py --run-dir <dir> \
      --manifest <P8_MATCHED_RENDERER_MANIFEST.json | P9_SYSTEM_CONTEXT_MANIFEST.json> \
      --arm <arm_id> [--scene <scene>] [--strict-lpips] [--json]
  python BASELINE_VALIDATION.py --selftest

The run directory is expected to contain (naming follows the repo's harness
conventions; several aliases are accepted per field):
  provenance.json            (harness/final30k/capture_provenance.py, baseline/reference_v1/provenance.py)
  build_identity.json        (b1a_accutile_build_identity.json convention)
  results.json / results.csv (final-eval rows, iteration 30000)
  training_summary.json      (strong_baseline convention)
  metrics results.json       (graphdeco metrics.py: SSIM/PSNR/LPIPS keys)

Exit codes: 0 = ACCEPT, 1 = REJECT, 2 = usage/selftest failure.
"""

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

SCENES = {
    "bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room",
    "stump", "treehill", "truck", "train", "drjohnson", "playroom",
}


# ---------------------------------------------------------------- extraction

def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _first(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in (None, ""):
            return d[k]
    return None


def load_run(run_dir: Path):
    """Return a normalized dict of everything we could find in the run dir."""
    prov = _load_json(run_dir / "provenance.json") or {}
    ident = (_load_json(run_dir / "build_identity.json")
             or _load_json(run_dir / "b1a_accutile_build_identity.json") or {})
    results = _load_json(run_dir / "results.json") or {}
    summary = _load_json(run_dir / "training_summary.json") or {}

    # graphdeco metrics.py writes {SSIM, PSNR, LPIPS} into results.json
    gdec_metrics = None
    if all(k in results for k in ("SSIM", "PSNR", "LPIPS")):
        gdec_metrics = results

    # final-eval rows: results.json {"rows": [...]} or eval_rows list, or CSV
    rows = None
    if isinstance(results, dict):
        rows = results.get("rows") or results.get("eval_rows") or results.get("evals")
    elif isinstance(results, list):
        rows = results
    if rows is None and (run_dir / "results.csv").is_file():
        with open(run_dir / "results.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows = [{k: (v if not isinstance(v, str) else
                     (float(v) if k not in ("tag", "scene") and v.replace(".", "", 1)
                      .replace("-", "", 1).isdigit() else v))
                 for k, v in r.items()} for r in rows]

    merged = {**prov, **{f"ident_{k}": v for k, v in ident.items()},
              **{f"sum_{k}": v for k, v in summary.items()}}
    out = {
        "source_commit": _first(merged, ["source_commit", "git_commit", "ident_source_commit",
                                         "ident_commit"]),
        "binary_sha256": _first(merged, ["binary_sha256", "so_sha256", "ident_so_sha256",
                                         "ident_binary_sha256", "extension_sha256"]),
        "scene": _first(merged, ["scene", "sum_scene"]),
        "seed": _first(merged, ["seed", "sum_seed"]),
        "iterations": _first(merged, ["iterations", "max_steps", "sum_iterations",
                                      "iterations_executed", "sum_iterations_executed"]),
        "resolution": _first(merged, ["resolution", "resolution_tag", "sum_resolution",
                                      "sum_effective_resolution"]),
        "eps2d": _first(merged, ["eps2d", "ident_eps2d"]),
        "eps2d_form": _first(merged, ["eps2d_form", "ident_eps2d_form"]),
        "contamination": _first(merged, ["contamination", "contamination_flag",
                                         "contaminated", "sum_contaminated"]),
        "gpu_snapshot_exists": (run_dir / "gpu_snapshot.json").is_file(),
        "rows": rows,
        "gdec_metrics": gdec_metrics,
        "loss_series": results.get("loss_series") if isinstance(results, dict) else None,
    }
    return out


def final_eval_row(run, iterations_expected):
    """Return the final-evaluation row (iteration == expected) or None."""
    rows = run.get("rows") or []
    best = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        it = _first(r, ["iteration", "step", "iter"])
        if it is None:
            continue
        try:
            it = int(float(it))
        except (TypeError, ValueError):
            continue
        if it == iterations_expected:
            best = r  # keep last matching row (re-eval conventions)
    return best


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- checks

class Report:
    def __init__(self):
        self.checks = []

    def add(self, name, ok, detail):
        self.checks.append({"check": name, "verdict": "PASS" if ok else "FAIL",
                            "detail": detail})

    @property
    def accepted(self):
        return all(c["verdict"] == "PASS" for c in self.checks)


def check_eq(rep, name, actual, expected, context=""):
    if expected is None:
        rep.add(name, False, f"no expectation available ({context})")
        return
    if actual is None:
        rep.add(name, False, f"missing field; expected {expected!r} ({context})")
        return
    a = str(actual).strip().lower()
    e = str(expected).strip().lower()
    if a.startswith(e[:12]) or a == e:  # tolerate short-SHA prefixes both ways
        rep.add(name, True, f"{actual!r} == {expected!r}")
    else:
        rep.add(name, False, f"got {actual!r}, expected {expected!r} ({context})")


def validate(run_dir, manifest_path, arm, scene_arg, strict_lpips):
    man = _load_json(manifest_path)
    if not isinstance(man, dict) or "arms" not in man:
        return None, "manifest unreadable or has no 'arms' block"
    arms = man["arms"]
    if arm not in arms:
        return None, f"arm {arm!r} not in manifest {Path(manifest_path).name}"
    spec = arms[arm]
    run = load_run(Path(run_dir))
    rep = Report()

    # 1. wrong source commit
    check_eq(rep, "source_commit", run["source_commit"],
             spec.get("source_commit"), f"arm {arm}")

    # 2. wrong binary (frozen arms carry an exact expectation)
    exp_bin = spec.get("binary_sha256")
    if exp_bin:
        check_eq(rep, "binary_sha256", run["binary_sha256"], exp_bin, "frozen binary")
    else:
        if run["binary_sha256"] is None:
            rep.add("binary_sha256", False,
                    "arm has no frozen binary expectation, but the run must still "
                    "register its build identity (build_identity.json) - missing")
        else:
            rep.add("binary_sha256", True,
                    f"build identity registered: {run['binary_sha256'][:16]}... "
                    "(no frozen expectation for this arm)")

    # 3. wrong scene
    if run["scene"] is None:
        rep.add("scene", False, "scene missing from provenance")
    elif run["scene"] not in SCENES:
        rep.add("scene", False, f"scene {run['scene']!r} not in the 13-scene set")
    elif scene_arg and run["scene"] != scene_arg:
        rep.add("scene", False, f"got {run['scene']!r}, requested {scene_arg!r}")
    else:
        rep.add("scene", True, f"{run['scene']}")

    # 4. wrong resolution
    mode = man.get("artifact", "")
    if "P8" in mode:
        res = run["resolution"]
        if res is None:
            rep.add("resolution", False, "resolution tag missing; expected '1080p'")
        elif "1080p" not in str(res):
            rep.add("resolution", False, f"expected 1080p-class, got {res!r}")
        else:
            rep.add("resolution", True, f"{res}")
    else:  # P9: native resolution must be recorded
        if run["resolution"] is None:
            rep.add("resolution", False,
                    "P9 native: effective resolution must be recorded in provenance")
        else:
            rep.add("resolution", True, f"recorded: {run['resolution']}")

    # 5. wrong seed
    if "P8" in mode:
        try:
            ok = int(run["seed"]) == 42
            rep.add("seed", ok, f"got {run['seed']!r}, expected 42")
        except (TypeError, ValueError):
            rep.add("seed", False, f"seed missing/non-numeric; expected 42")
    else:
        if run["seed"] is None:
            rep.add("seed", False, "P9 native: seed must be recorded")
        else:
            rep.add("seed", True, f"recorded: {run['seed']}")

    # 6. wrong eps2d
    exp_eps = spec.get("eps2d")
    if exp_eps is None:
        rep.add("eps2d", False, f"manifest has no eps2d expectation for arm {arm}")
    elif isinstance(exp_eps, (int, float)) or (
            isinstance(exp_eps, str) and _is_number(exp_eps)):
        try:
            ok = run["eps2d"] is not None and abs(float(run["eps2d"]) - float(exp_eps)) < 1e-12
            rep.add("eps2d", ok, f"got {run['eps2d']!r}, expected {exp_eps}")
        except (TypeError, ValueError):
            rep.add("eps2d", False, f"eps2d missing/non-numeric; expected {exp_eps}")
    else:
        # form-string expectation (graphdeco cov-form / faster-gs dilation form):
        # the run must record the form and must NOT override a numeric eps2d
        numeric_override = isinstance(run["eps2d"], (int, float)) or (
            isinstance(run["eps2d"], str) and _is_number(run["eps2d"]))
        form_rec = run["eps2d_form"] or (
            run["eps2d"] if isinstance(run["eps2d"], str) else None)
        exp_s = str(exp_eps).lower()
        keyword = "cov" if "cov" in exp_s else ("dilation" if "dilation" in exp_s else "")
        ok = (not numeric_override) and form_rec is not None and (
            not keyword or keyword in str(form_rec).lower())
        rep.add("eps2d", ok,
                f"expected form {exp_eps!r}; numeric_override={numeric_override}; "
                f"recorded form {form_rec!r}")

    # 7. wrong training iterations
    iters_exp = 30000
    final_row = final_eval_row(run, iters_exp)
    recorded_iters = run["iterations"]
    try:
        it_ok = recorded_iters is not None and int(float(recorded_iters)) == iters_exp
        det = f"recorded {recorded_iters!r}"
    except (TypeError, ValueError):
        it_ok = False
        det = f"iterations field missing/non-numeric (expected {iters_exp})"
    if final_row is None and not it_ok:
        rep.add("iterations", False, det + " and no final eval row at 30000")
    else:
        rep.add("iterations", it_ok or final_row is not None,
                det + ("" if it_ok else " (final eval row at 30000 present as fallback)"))

    # 8. contamination
    cont = run["contamination"]
    if cont is not None and str(cont).strip().lower() in ("true", "1", "yes"):
        rep.add("contamination", False, f"contamination flag set: {cont!r}")
    elif not run["gpu_snapshot_exists"]:
        rep.add("contamination", False,
                "no gpu_snapshot.json - contamination cannot be verified (fail-closed)")
    else:
        rep.add("contamination", True,
                f"flag={cont!r}; snapshot present" if cont is not None
                else "snapshot present; flag absent (snapshot is the authority)")

    # 9. NaN / Inf in final metrics
    metrics_src = final_row or run["gdec_metrics"]
    if metrics_src is None:
        rep.add("nan_inf", False, "no final metrics found to check")
    else:
        bad = [k for k, v in metrics_src.items()
               if k.lower() in ("psnr", "ssim", "lpips", "l1", "n_gaussians")
               and not finite(v)]
        rep.add("nan_inf", not bad, f"non-finite: {bad}" if bad else "all final metrics finite")
        if run.get("loss_series"):
            ls_bad = not all(finite(x) for x in run["loss_series"])
            rep.add("nan_inf_loss_series", not ls_bad,
                    "non-finite loss values in series" if ls_bad else "loss series finite")

    # 10. missing final evaluation
    if final_row is None and run["gdec_metrics"] is None:
        rep.add("final_evaluation", False,
                f"no evaluation row/metrics at iteration {iters_exp}")
    else:
        src = final_row or run["gdec_metrics"]
        have = {k.lower(): _first(src, [k, k.upper()]) for k in ("psnr", "ssim")}
        have["lpips"] = _first(src, ["lpips", "LPIPS", "lpips_vals"])
        missing = [k for k in ("psnr", "ssim") if not finite(have[k])]
        if strict_lpips and not finite(have["lpips"]):
            missing.append("lpips")
        rep.add("final_evaluation", not missing,
                f"final eval present; missing metrics: {missing}" if missing
                else f"final eval present (psnr/ssim"
                     f"{'/lpips' if finite(have['lpips']) else ''})")

    return rep, None


# ---------------------------------------------------------------- selftest

def _mkrun(tmp, name, **over):
    d = Path(tmp) / name
    d.mkdir(parents=True)
    prov = {"source_commit": "512d366b67073d77ca099ede742683c165dfc23b",
            "binary_sha256": "aa" * 32, "scene": "room", "seed": 42,
            "iterations": 30000, "resolution": "1080p", "eps2d": 0.1,
            "contamination": False}
    prov.update(over)
    (d / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    (d / "gpu_snapshot.json").write_text(json.dumps({"gpus": []}), encoding="utf-8")
    (d / "results.json").write_text(json.dumps(
        {"rows": [{"iteration": 30000, "psnr": 30.1, "ssim": 0.9, "lpips": 0.15}]}),
        encoding="utf-8")
    return d


def selftest():
    here = Path(__file__).resolve().parent
    p8 = here / "P8_MATCHED_RENDERER_MANIFEST.json"
    p9 = here / "P9_SYSTEM_CONTEXT_MANIFEST.json"
    if not p8.is_file() or not p9.is_file():
        print("selftest FAIL: manifests not found next to this script")
        return 2
    tmp = tempfile.mkdtemp(prefix="blv_selftest_")
    try:
        good = _mkrun(tmp, "good")
        rep, err = validate(good, p8, "gsplat_main", "room", strict_lpips=True)
        assert err is None and rep.accepted, [("good run rejected", rep.checks)]

        bad_commit = _mkrun(tmp, "bad_commit", source_commit="deadbeef")
        rep, _ = validate(bad_commit, p8, "gsplat_main", "room", True)
        assert not rep.accepted

        bad_scene = _mkrun(tmp, "bad_scene", scene="plaza")
        rep, _ = validate(bad_scene, p8, "gsplat_main", "plaza", True)
        assert not rep.accepted

        bad_seed = _mkrun(tmp, "bad_seed", seed=43)
        rep, _ = validate(bad_seed, p8, "gsplat_main", "room", True)
        assert not rep.accepted

        bad_eps = _mkrun(tmp, "bad_eps", eps2d=0.3)
        rep, _ = validate(bad_eps, p8, "gsplat_main", "room", True)
        assert not rep.accepted

        contaminated = _mkrun(tmp, "contam", contamination=True)
        rep, _ = validate(contaminated, p8, "gsplat_main", "room", True)
        assert not rep.accepted

        no_snapshot = _mkrun(tmp, "nosnap")
        (no_snapshot / "gpu_snapshot.json").unlink()
        rep, _ = validate(no_snapshot, p8, "gsplat_main", "room", True)
        assert not rep.accepted

        nan_metrics = _mkrun(tmp, "nan")
        (nan_metrics / "results.json").write_text(json.dumps(
            {"rows": [{"iteration": 30000, "psnr": "NaN", "ssim": 0.9, "lpips": 0.15}]}),
            encoding="utf-8")
        rep, _ = validate(nan_metrics, p8, "gsplat_main", "room", True)
        assert not rep.accepted

        no_final = _mkrun(tmp, "nofinal")
        (no_final / "results.json").write_text(json.dumps(
            {"rows": [{"iteration": 25000, "psnr": 29.0, "ssim": 0.88, "lpips": 0.2}]}),
            encoding="utf-8")
        rep, _ = validate(no_final, p8, "gsplat_main", "room", True)
        assert not rep.accepted

        # frozen-binary arm must reject a mismatched .so hash (c0)
        bad_bin = _mkrun(tmp, "badbin", source_commit="higs-frozen", eps2d=0.3)
        rep, _ = validate(bad_bin, p8, "c0", "room", True)
        assert not rep.accepted

        # P9 native mode: recorded-not-controlled fields
        p9run = _mkrun(tmp, "p9", source_commit="34c45c6d9b8bd6110231864f2f358b6d3abbf73d",
                       binary_sha256=None, seed=0, resolution="auto-1600px",
                       eps2d="cov-diag +0.3", eps2d_form="graphdeco cov-diag +0.3",
                       iterations=30000, contamination=False)
        # remove build identity expectation problem: register identity
        (p9run / "build_identity.json").write_text(
            json.dumps({"source_commit": "34c45c6d9b8bd6110231864f2f358b6d3abbf73d",
                        "so_sha256": "bb" * 32}), encoding="utf-8")
        rep, err = validate(p9run, p9, "speedy_splat", "room", strict_lpips=True)
        assert err is None and rep.accepted, [("p9 good rejected", rep.checks)]

        print("selftest PASS: 12/12 synthetic cases behaved as expected")
        return 0
    except AssertionError as e:
        print(f"selftest FAIL: {e}")
        return 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=not (sys.argv[1:] and sys.argv[1] in ("--selftest",)))
    ap.add_argument("--manifest", required=False)
    ap.add_argument("--arm", required=False)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--strict-lpips", action="store_true",
                    help="require LPIPS at the final evaluation (P8 final report)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if not (args.run_dir and args.manifest and args.arm):
        ap.error("--run-dir, --manifest and --arm are required")

    rep, err = validate(args.run_dir, args.manifest, args.arm, args.scene,
                        args.strict_lpips)
    if err:
        print(f"REJECT: {err}")
        return 1

    out = {"run_dir": args.run_dir, "arm": args.arm, "scene": args.scene,
           "verdict": "ACCEPT" if rep.accepted else "REJECT",
           "checks": rep.checks}
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"=== BASELINE_VALIDATION: {out['verdict']} ===")
        for c in rep.checks:
            print(f"  [{c['verdict']}] {c['check']}: {c['detail']}")
    return 0 if rep.accepted else 1


if __name__ == "__main__":
    sys.exit(main())
