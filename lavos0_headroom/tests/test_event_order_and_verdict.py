"""Phase A event-ordering + verdict/summary integration tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lavos0 import event_order                      # noqa: E402
from lavos0 import verdict                          # noqa: E402
from lavos0 import summary                          # noqa: E402


def test_optimizer_step_after_topology():
    assert event_order.optimizer_step_after_topology() is True


def test_topology_sub_order():
    assert event_order.topology_sub_order() == [
        "densify_and_clone", "densify_and_split", "prune_points"]


def test_phase_a_gate_resolves():
    res = event_order.phase_a_gate(max_iterations=15000)
    assert res.resolved is True
    assert res.ordering


def test_event_trace_topology_present():
    trace = event_order.event_order_trace(
        500, 15000, 3000, 100, 15000)
    topo = [r for r in trace if r["event"] == "topology"]
    assert len(topo) > 0
    # optimizer step is a no-op on topology iterations
    assert all(r["optimizer_step_is_noop"] for r in topo)


def test_verdict_precedence():
    assert verdict.fold_verdict({"source_identity_pass": False})["verdict"] \
        == "LAVOS0_SOURCE_IDENTITY_FAIL"
    assert verdict.fold_verdict({
        "source_identity_pass": True,
        "event_semantics_resolved": False})["verdict"] \
        == "LAVOS0_EVENT_SEMANTICS_UNRESOLVED"
    assert verdict.fold_verdict({
        "source_identity_pass": True, "event_semantics_resolved": True,
        "structural_headroom_pass": False})["verdict"] \
        == "LAVOS0_NO_STRUCTURAL_HEADROOM"
    assert verdict.fold_verdict({
        "source_identity_pass": True, "event_semantics_resolved": True,
        "structural_headroom_pass": True,
        "topology_exactness_pass": False})["verdict"] \
        == "LAVOS0_TOPOLOGY_EXACTNESS_FAIL"
    assert verdict.fold_verdict({
        "source_identity_pass": True, "event_semantics_resolved": True,
        "structural_headroom_pass": True,
        "topology_exactness_pass": True,
        "performance_headroom_pass": False})["verdict"] \
        == "LAVOS0_PERFORMANCE_HEADROOM_FAIL"


def test_verdict_promote_and_strong():
    base = {
        "source_identity_pass": True, "event_semantics_resolved": True,
        "structural_headroom_pass": True, "topology_exactness_pass": True,
        "performance_headroom_pass": True, "strong_headroom_pass": False}
    assert verdict.fold_verdict(base)["verdict"] == "LAVOS0_HEADROOM_PROMOTE"
    base["strong_headroom_pass"] = True
    assert verdict.fold_verdict(base)["verdict"] == "LAVOS0_STRONG_HEADROOM_PROMOTE"
    assert verdict.is_promote("LAVOS0_HEADROOM_PROMOTE")
    assert not verdict.is_promote("LAVOS0_NO_STRUCTURAL_HEADROOM")


def test_summary_fail_closed_when_empty():
    s = summary.build_summary()
    assert s["source_identity_pass"] is False
    assert s["structural_headroom_pass"] is False
    v = verdict.fold_verdict(s)
    assert v["verdict"] == "LAVOS0_SOURCE_IDENTITY_FAIL"


def test_summary_full_pipeline_promote():
    headroom = {"passed": True, "aggregate": {"median_avoidable_transition_fraction": 0.6},
                "per_scene": {"room": {"avoidable_transition_fraction": 0.6}}}
    perf = {"performance_headroom_pass": True, "strong_headroom_pass": True,
            "per_scene": {"room": {"optimizer_stage_speedup": 1.8}},
            "geomean_optimizer_speedup": 1.9}
    topo = {"exactness_pass": True, "n_fences": 4, "n_mismatches": 0}
    s = summary.build_summary(
        source_identity={"source_identity_pass": True},
        event_order={"resolved": True, "notes": "ok"},
        headroom=headroom, topology_oracle=topo, perf_oracle=perf)
    assert s["source_identity_pass"] is True
    assert s["structural_headroom_pass"] is True
    assert s["topology_exactness_pass"] is True
    assert s["performance_headroom_pass"] is True
    assert s["strong_headroom_pass"] is True
    v = verdict.fold_verdict(s)
    assert v["verdict"] == "LAVOS0_STRONG_HEADROOM_PROMOTE"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL", len(fns), "event_order/verdict tests passed")
