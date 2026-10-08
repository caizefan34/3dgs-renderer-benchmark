"""Authoritatively recompute P12 paired statistics from raw 13-scene runs."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

from p12_pair_manifest import CAMPAIGNS
from p12_run_guard import SCENES


T_CRITICAL_DF12_975 = 2.1788128296634177
CI_METHOD = "two-sided Student t interval on paired log wall ratios; n=13, df=12, t(0.975)=2.1788128296634177, sample SD (ddof=1)"
TOLERANCE = 5e-5  # permits a four-decimal published GM, not a wrong formula


def geometric_mean(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("GM requires finite positive ratios")
    return math.exp(statistics.mean(math.log(x) for x in values))


def wall_reduction(gm):
    if not math.isfinite(gm) or gm <= 0:
        raise ValueError("GM must be finite and positive")
    return 1 - 1 / gm


def log_ratio_ci(ratios):
    if len(ratios) != 13:
        raise ValueError(f"95% CI requires 13 valid scenes, got {len(ratios)}")
    logs = [math.log(value) for value in ratios]
    center = statistics.mean(logs)
    margin = T_CRITICAL_DF12_975 * statistics.stdev(logs) / math.sqrt(13)
    return [center - margin, center + margin]


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denominator = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denominator if denominator > 0 else None


def _finite_positive(value, label):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: missing or nonnumeric") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label}: must be finite and positive")
    return value


def recompute(raw, tolerance=TOLERANCE):
    campaign = raw.get("campaign")
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown campaign {campaign!r}")
    base_arm, test_arm = CAMPAIGNS[campaign]
    runs = raw.get("runs")
    if not isinstance(runs, list):
        raise ValueError("raw P12 JSON must contain a runs list")
    clean = [row for row in runs if row.get("status") == "clean" and row.get("contaminated") is False]
    contaminated = [row for row in runs if row.get("status") != "clean" or row.get("contaminated") is not False]
    if len(clean) != 26:
        raise ValueError(f"valid-scene count: expected 13 complete pairs / 26 runs, got {len(clean)} runs")
    by_key = {}
    for row in clean:
        key = (row.get("scene"), row.get("arm"))
        if key in by_key:
            raise ValueError(f"duplicate run {key}")
        by_key[key] = row
    scenes = sorted(SCENES)
    if set(row.get("scene") for row in clean) != SCENES:
        raise ValueError(f"valid-scene count: missing or unknown scenes; expected {scenes}")
    per_scene, ratios, n_logs = [], [], []
    deltas = {metric: [] for metric in ("psnr", "ssim", "lpips")}
    for scene in scenes:
        if (scene, base_arm) not in by_key or (scene, test_arm) not in by_key:
            raise ValueError(f"{scene}: missing {base_arm} or {test_arm}")
        base, test = by_key[(scene, base_arm)], by_key[(scene, test_arm)]
        wall_base = _finite_positive(base.get("wall_time_s"), f"{scene}/{base_arm} wall_time_s")
        wall_test = _finite_positive(test.get("wall_time_s"), f"{scene}/{test_arm} wall_time_s")
        ratio = wall_base / wall_test
        ratios.append(ratio)
        n_base = _finite_positive(base.get("final_n"), f"{scene}/{base_arm} final_n")
        n_test = _finite_positive(test.get("final_n"), f"{scene}/{test_arm} final_n")
        n_ratio = n_test / n_base
        n_logs.append(math.log(n_ratio))
        quality = {}
        for metric in deltas:
            bval, tval = float(base[metric]), float(test[metric])
            if not math.isfinite(bval) or not math.isfinite(tval):
                raise ValueError(f"{scene}: nonfinite {metric}")
            # P12-A protocol specifies GMAIN - C0; P12-B specifies A1 - A0.
            delta = bval - tval if campaign == "P12-A" else tval - bval
            quality[metric] = delta
            deltas[metric].append(delta)
        per_scene.append({"scene": scene, "wall_ratio": ratio, "final_n_ratio_test_over_base": n_ratio, "quality_delta": quality})
    gm = geometric_mean(ratios)
    reduction = wall_reduction(gm)
    reported = raw.get("reported")
    if not isinstance(reported, dict) or "gm" not in reported or "wall_reduction" not in reported:
        raise ValueError("reported.gm and reported.wall_reduction are required for consistency checks")
    if not math.isclose(float(reported["gm"]), gm, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"reported GM {reported['gm']} differs from raw recomputation {gm}")
    if not math.isclose(float(reported["wall_reduction"]), reduction, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"reported wall reduction {reported['wall_reduction']} differs from 1 - 1/GM = {reduction}")
    ci = log_ratio_ci(ratios)
    return {
        "campaign": campaign, "valid_scene_count": 13, "excluded_run_count": len(contaminated), "base_arm": base_arm, "test_arm": test_arm,
        "per_scene": per_scene, "gm": gm, "wall_reduction": reduction,
        "median_ratio": statistics.median(ratios), "min_ratio": min(ratios), "max_ratio": max(ratios),
        "positive_count": sum(r > 1 for r in ratios), "mean_log_ratio": statistics.mean(math.log(r) for r in ratios),
        "ci95_mean_log_ratio": ci, "ci95_ratio": [math.exp(x) for x in ci], "ci95_method": CI_METHOD,
        "quality_delta_definition": "base - test" if campaign == "P12-A" else "test - base",
        "quality_delta": {k: {"mean": statistics.mean(v), "median": statistics.median(v)} for k, v in deltas.items()},
        "pearson_log_wall_vs_log_final_n_ratio": _pearson([math.log(r) for r in ratios], n_logs),
        "final_n_ratio_definition": "test / base", "consistency": "PASS",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = recompute(json.loads(args.input.read_text(encoding="utf-8")))
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        print(f"P12 stats invariant violation: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("gm", "wall_reduction", "valid_scene_count", "ci95_method")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
