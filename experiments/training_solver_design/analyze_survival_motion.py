"""Check source identities and summarize paired 3D motion at the fixed state."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).parent
FILES = (
    ("room_survival_motion_v1.json", {"survival_pair_motion_probe.py": "survival_pair_motion_probe_v1_failed.py"}),
    ("room_survival_motion_v2.json", {"survival_pair_motion_probe.py": "survival_pair_motion_probe_v2.py"}),
    ("room_survival_motion_v3_B.json", {}),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = {}
    for filename, source_aliases in FILES:
        payload = json.loads((ROOT / filename).read_text(encoding="utf-8"))
        assert payload["completed"], filename
        for source, expected in payload["sources"].items():
            actual = sha(ROOT / source_aliases.get(source, source))
            assert actual == expected, (filename, source, expected, actual)
        records[filename] = payload
    initial = records["room_survival_motion_v1.json"]
    a = records["room_survival_motion_v2.json"]
    b = records["room_survival_motion_v3_B.json"]
    assert a["checkpoint_sha256"] == b["checkpoint_sha256"] == initial["checkpoint_sha256"]
    assert a["n_selected"] == b["n_selected"] == 256
    assert len(a["views"]) == len(b["views"]) == 3
    assert not set(a["view_ids"]) & set(b["view_ids"])
    assert a["shifts_sigma"] == b["shifts_sigma"] == [0., .1, .3, .45]
    # The first run had out-of-view offspring still contributing through an old
    # parent tile. Preserve it as a diagnosed failure, never average it in.
    assert max(s["group_vs_baseline"]["max_abs"]
               for v in initial["views"] for s in v["shifts"]) > 10
    assert max(s["group_vs_baseline"]["max_abs"]
               for d in (a, b) for v in d["views"] for s in v["shifts"]) < 1
    for d in (a, b):
        for v in d["views"]:
            assert [s["shift_sigma"] for s in v["shifts"]] == d["shifts_sigma"]
            assert all(s["coverage_and_order"]["tile_coverage_missing"] == 0 for s in v["shifts"])
            assert v["shifts"][0]["coverage_and_order"]["new_union_group_tile_entries"] == 0
            assert v["shifts"][0]["group_vs_baseline"]["max_abs"] < 1e-6

    summary = {"scope": "one room checkpoint, 256 fixed parents, six training cameras, no update or official unseen view",
               "source_identity_checks_passed": True,
               "input_sha256": {name: sha(ROOT / name) for name in records},
               "analyzer_sha256": sha(Path(__file__)),
               "preserved_first_run_visibility_failure_max_image_delta":
                   max(s["group_vs_baseline"]["max_abs"]
                       for v in initial["views"] for s in v["shifts"]),
               "checkpoint_sha256": a["checkpoint_sha256"],
               "candidate_count": 256,
               "groups": {},
               "trainable_motion_and_exit_verified": False,
               "full_training_quality_or_speed_verified": False}
    for label, d in (("A", a), ("B", b)):
        group = {"view_ids": d["view_ids"], "by_shift": []}
        for j, shift in enumerate(d["shifts_sigma"]):
            arms = [v["shifts"][j] for v in d["views"]]
            group["by_shift"].append({
                "shift_sigma": shift,
                "group_loss_delta_mean": statistics.mean(x["group_loss_delta"] for x in arms),
                "physical_survival_loss_delta_mean": statistics.mean(x["physical_loss_delta"] for x in arms),
                "matched_sixty_percent_q_loss_delta_mean": statistics.mean(
                    x["matched_geometry_standard_controls"]["sixty_percent_q"]["loss_delta"] for x in arms),
                "interleaved_tile_fraction_min": min(x["coverage_and_order"]["interleaving_fraction"] for x in arms),
                "interleaved_tile_fraction_max": max(x["coverage_and_order"]["interleaving_fraction"] for x in arms),
                "group_vs_physical_max_rgb": max(x["group_vs_physical"]["max_abs"] for x in arms),
                "group_union_extra_tile_entries_total": sum(
                    x["coverage_and_order"]["new_union_group_tile_entries"] for x in arms),
                "per_view_group_loss_delta": [x["group_loss_delta"] for x in arms],
                "per_view_physical_loss_delta": [x["physical_loss_delta"] for x in arms],
            })
        summary["groups"][label] = group
    out = ROOT / "survival_motion_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({label: [{k: x[k] for k in (
        "shift_sigma", "group_loss_delta_mean", "physical_survival_loss_delta_mean",
        "interleaved_tile_fraction_min", "interleaved_tile_fraction_max",
        "group_vs_physical_max_rgb")}
        for x in group["by_shift"]] for label, group in summary["groups"].items()}, indent=2))


if __name__ == "__main__":
    main()
