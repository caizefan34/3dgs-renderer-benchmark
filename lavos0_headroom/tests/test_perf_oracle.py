"""Phase E performance-oracle + Amdahl tests (PREREG Sec.7-8)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lavos0 import perf_oracle                      # noqa: E402
from lavos0 import config                           # noqa: E402


def test_amdahl_speedup():
    assert perf_oracle.amdahl_optimizer_speedup(0.5) == 2.0
    assert abs(perf_oracle.amdahl_optimizer_speedup(0.25) - 4.0) < 1e-9
    # degenerate: full consumption => ~1x
    assert abs(perf_oracle.amdahl_optimizer_speedup(1.0) - 1.0) < 1e-9


def test_geomean():
    gm = perf_oracle.geometric_mean([2.0, 8.0])
    assert abs(gm - 4.0) < 1e-9
    assert perf_oracle.geometric_mean([]) == 0.0


def test_scene_perf_pass():
    # consumer fraction 0.5 => 2x speedup; large wall; tiny overhead
    p = perf_oracle.ScenePerf(
        scene="room", consumer_fraction=0.5, sh_adam_wall_fraction=0.2,
        t_pre_wall=10000.0, t_post_wall=8000.0,
        dense_sh_adam_ms_per_iter=10.0, lazy_sh_adam_ms_per_iter=5.0,
        forced_settle_overhead=0.1)
    assert p.optimizer_stage_speedup() >= config.PERF_MIN_SPEEDUP_EVERY
    assert p.incremental_full30k_reduction_fraction() > 0
    assert p.all_pass()


def test_scene_perf_regression_fails():
    # overhead exceeds gross saving => net negative, gate fails
    p = perf_oracle.ScenePerf(
        scene="room", consumer_fraction=0.9, sh_adam_wall_fraction=0.2,
        t_pre_wall=10000.0, t_post_wall=8000.0,
        dense_sh_adam_ms_per_iter=10.0, lazy_sh_adam_ms_per_iter=9.0,
        forced_settle_overhead=5.0)
    assert p.overhead_fraction_of_gross() > config.PERF_MAX_OVERHEAD_FRACTION
    assert not p.all_pass()


def test_evaluate_performance_aggregate():
    good = perf_oracle.ScenePerf(
        "room", 0.5, 0.2, 10000.0, 8000.0, 10.0, 5.0, 0.1)
    good2 = perf_oracle.ScenePerf(
        "garden", 0.5, 0.2, 10000.0, 8000.0, 10.0, 5.0, 0.1)
    good3 = perf_oracle.ScenePerf(
        "bicycle", 0.5, 0.2, 10000.0, 8000.0, 10.0, 5.0, 0.1)
    res = perf_oracle.evaluate_performance(
        {"room": good, "garden": good2, "bicycle": good3})
    assert res["every_scene_speedup_ge_1.35"] is True
    assert res["performance_headroom_pass"] is True
    assert res["geomean_optimizer_speedup"] >= config.PERF_GEOMEAN_SPEEDUP


def test_evaluate_performance_fails_one_scene_speedup():
    a = perf_oracle.ScenePerf("room", 0.9, 0.2, 10000.0, 8000.0, 10.0, 9.5, 0.0)
    b = perf_oracle.ScenePerf("garden", 0.5, 0.2, 10000.0, 8000.0, 10.0, 5.0, 0.1)
    c = perf_oracle.ScenePerf("bicycle", 0.5, 0.2, 10000.0, 8000.0, 10.0, 5.0, 0.1)
    res = perf_oracle.evaluate_performance({"room": a, "garden": b, "bicycle": c})
    assert res["every_scene_speedup_ge_1.35"] is False
    assert res["performance_headroom_pass"] is False


def test_microbench_speedup_caps_amdahl():
    # if microbench says 1.2x but amdahl says 3x, use the lower (1.2x)
    p = perf_oracle.ScenePerf(
        scene="room", consumer_fraction=0.33, sh_adam_wall_fraction=0.2,
        t_pre_wall=10000.0, t_post_wall=8000.0,
        dense_sh_adam_ms_per_iter=10.0, lazy_sh_adam_ms_per_iter=8.0,
        forced_settle_overhead=0.0, microbench_speedup=1.2)
    assert abs(p.optimizer_stage_speedup() - 1.2) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL", len(fns), "perf_oracle tests passed")
