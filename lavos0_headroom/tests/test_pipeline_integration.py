"""End-to-end integration: SUMMARY -> frozen analyzer -> VERDICT.

Uses the byte-identical frozen analyzer (lavos0_analyze.py) copied from the
packet. NOTE: the frozen analyzer reads sys.argv[1] and writes the verdict at
IMPORT TIME (it has no main() guard), so sys.argv must be set before exec_module.
"""

import importlib.util
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from lavos0 import summary              # noqa: E402


def _run_analyzer(summary_obj):
    path = os.path.join(ROOT, "lavos0_analyze.py")
    with tempfile.TemporaryDirectory() as d:
        spath = os.path.join(d, "LAVOS0_SUMMARY.json")
        with open(spath, "w", encoding="utf-8") as f:
            json.dump(summary_obj, f)
        argv_backup = sys.argv
        sys.argv = ["lavos0_analyze.py", spath]
        try:
            spec = importlib.util.spec_from_file_location("lavos0_analyze", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)   # analyzer runs at import time
        finally:
            sys.argv = argv_backup
        with open(os.path.join(d, "LAVOS0_VERDICT.json"), encoding="utf-8") as f:
            return json.load(f)


def _promote_summary(strong):
    headroom = {"passed": True,
                "aggregate": {"median_avoidable_transition_fraction": 0.6},
                "per_scene": {"room": {"avoidable_transition_fraction": 0.6}}}
    perf = {"performance_headroom_pass": True, "strong_headroom_pass": strong,
            "per_scene": {"room": {"optimizer_stage_speedup": 1.8}},
            "geomean_optimizer_speedup": 1.9}
    topo = {"exactness_pass": True, "n_fences": 4, "n_mismatches": 0}
    return summary.build_summary(
        source_identity={"source_identity_pass": True},
        event_order={"resolved": True, "notes": "ok"},
        headroom=headroom, topology_oracle=topo, perf_oracle=perf)


def test_pipeline_promote():
    v = _run_analyzer(_promote_summary(strong=False))
    assert v["verdict"] == "LAVOS0_HEADROOM_PROMOTE", v


def test_pipeline_strong_promote():
    v = _run_analyzer(_promote_summary(strong=True))
    assert v["verdict"] == "LAVOS0_STRONG_HEADROOM_PROMOTE", v


def test_pipeline_no_headroom():
    s = _promote_summary(strong=False)
    s["structural_headroom_pass"] = False
    v = _run_analyzer(s)
    assert v["verdict"] == "LAVOS0_NO_STRUCTURAL_HEADROOM", v


def test_pipeline_exactness_fail():
    s = _promote_summary(strong=False)
    s["topology_exactness_pass"] = False
    v = _run_analyzer(s)
    assert v["verdict"] == "LAVOS0_TOPOLOGY_EXACTNESS_FAIL", v


def test_pipeline_source_identity_fail():
    s = _promote_summary(strong=False)
    s["source_identity_pass"] = False
    v = _run_analyzer(s)
    assert v["verdict"] == "LAVOS0_SOURCE_IDENTITY_FAIL", v


def test_pipeline_perf_fail():
    s = _promote_summary(strong=False)
    s["performance_headroom_pass"] = False
    v = _run_analyzer(s)
    assert v["verdict"] == "LAVOS0_PERFORMANCE_HEADROOM_FAIL", v


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL", len(fns), "pipeline integration tests passed")
