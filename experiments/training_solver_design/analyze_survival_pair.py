"""Verify executed source identities and summarize the bounded CUDA audit."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs = {}
    for name, mapping in (
        ("room_survival_pair_v1_synthetic.json", {
            "survival_pair_probe.py": "survival_pair_probe_v1_failed.py",
            "survival_pair_cuda.cu": "survival_pair_cuda_v1.cu"}),
        ("room_survival_pair_v2.json", {"survival_pair_cuda.cu": "survival_pair_cuda_v1.cu"}),
        ("room_survival_pair_v3.json", {}),
    ):
        record = json.loads((ROOT / name).read_text(encoding="utf-8"))
        for source, expected in record["sources"].items():
            assert sha(ROOT / mapping.get(source, source)) == expected, (name, source)
        inputs[name] = record
    final = inputs["room_survival_pair_v3.json"]
    assert final["completed"] and final["synthetic_independent_child_vjp"]["passed"]
    assert not inputs["room_survival_pair_v1_synthetic.json"]["completed"]
    initial_full = inputs["room_survival_pair_v2.json"]
    assert initial_full["completed"]
    assert not initial_full["real_scene"]["views"][0]["arms"]["pair_all"]["image_gate"]
    assert all(r["native_sources"] == final["native_sources"] for r in inputs.values())
    views = final["real_scene"]["views"]
    arms = [a for v in views for a in v["arms"].values()]
    assert len(views) == 3 and len(arms) == 9
    assert all(a["image_gate"] and all(a["vjp_gate_by_parameter"].values()) for a in arms)
    summary = {
        "scope": "fixed-parent-list CUDA birth parity; no training or novelty proof",
        "source_identity_checks_passed": True,
        "input_sha256": {name: sha(ROOT / name) for name in inputs},
        "analyzer_sha256": sha(Path(__file__)),
        "synthetic_independent_child_vjp": final["synthetic_independent_child_vjp"],
        "preserved_initial_full_image_failure": initial_full["real_scene"]["views"][0]["arms"]["pair_all"]["image"],
        "full_scene": final["real_scene"]["scene"],
        "views": [v["camera_id"] for v in views],
        "configurations_passed": len(arms),
        "max_image_absolute_error": max(a["image"]["max_abs"] for a in arms),
        "max_alpha_absolute_error": max(a["alpha"]["max_abs"] for a in arms),
        "max_vjp_relative_l2": max(m["relative_l2"] for a in arms for m in a["vjp"].values()),
        "diagnostic_timing_medians_ms": {
            str(v["camera_id"]): {name: statistics.median(a["diagnostic_raster_loss_vjp_ms"])
                                  for name, a in v["arms"].items()} for v in views},
        "motion_coverage_and_ordering_validated": False,
        "training_benefit_verified": False,
        "novelty_established": False,
    }
    (ROOT / "survival_pair_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("scope", "source_identity_checks_passed", "configurations_passed",
                               "max_image_absolute_error", "max_alpha_absolute_error", "max_vjp_relative_l2")}, indent=2))


if __name__ == "__main__":
    main()
