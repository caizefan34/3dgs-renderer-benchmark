"""Phase B/C debt-replay + structural-headroom-gate tests (PREREG Sec.4-5)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lavos0 import debt_replay                      # noqa: E402
from lavos0 import consumer_trace                   # noqa: E402
from lavos0 import headroom_gates                   # noqa: E402
from lavos0 import config                           # noqa: E402


def test_all_consumed_no_debt():
    # every row consumed every step => no debt, required == dense
    consumers = {t: list(range(10)) for t in range(1, 11)}
    stats = debt_replay.replay_debt(consumers, n_iters=10, n_initial_rows=10)
    assert stats.consumer_settles == 0
    assert stats.topology_forced_settles == 0
    assert stats.dense_row_transitions == 100
    assert stats.required_row_transitions == 0
    assert stats.avoidable_fraction() == 1.0


def test_none_consumed_all_deferred():
    consumers = {t: [] for t in range(1, 11)}
    stats = debt_replay.replay_debt(consumers, n_iters=10, n_initial_rows=10)
    # never settled until a consumer/topology appears; here none => 0 settles
    assert stats.consumer_settles == 0
    assert stats.required_row_transitions == 0
    # dense still does 100 transitions
    assert stats.dense_row_transitions == 100


def test_consumer_clears_debt_and_counts():
    # rows 0..4 consumed at t=5 only; they accrue 4 debt then settle once
    consumers = {5: [0, 1, 2, 3, 4]}
    stats = debt_replay.replay_debt(consumers, n_iters=5, n_initial_rows=10)
    assert stats.consumer_settles == 5
    # debt lengths should be 4 each (settled at t=5, deferred t=1..4)
    assert sorted(stats.debt_lengths) == [4, 4, 4, 4, 4]
    assert stats.p50_debt == 4
    assert stats.max_debt == 4


def test_clone_forces_settle():
    # row 1 deferred t=1..3, cloned at t=4 => forced settle
    consumers = {}
    stats = debt_replay.replay_debt(
        consumers, n_iters=4, n_initial_rows=5,
        clone_parents={4: [1]})
    assert stats.topology_forced_settles == 1
    assert stats.consumer_settles == 0
    assert stats.rows_born == 1


def test_split_removes_parent_and_counts_children():
    consumers = {}
    stats = debt_replay.replay_debt(
        consumers, n_iters=2, n_initial_rows=4,
        split_parents={1: [2]})
    assert stats.topology_forced_settles == 0  # parent had no debt
    assert stats.rows_born == 2


def test_prune_discards_debt_without_settle():
    # row 3 deferred t=1..4 (debt 4), pruned at t=5 => discard, no settle
    consumers = {}
    stats = debt_replay.replay_debt(
        consumers, n_iters=5, n_initial_rows=6,
        prune_ids={5: [3]})
    assert stats.prune_discards == 1
    assert stats.consumer_settles == 0
    assert stats.topology_forced_settles == 0


def test_percentiles():
    consumers = {10: list(range(100))}
    stats = debt_replay.replay_debt(consumers, n_iters=10, n_initial_rows=100)
    # 100 rows deferred 9 steps then settled
    assert stats.max_debt == 9
    assert stats.p90_debt == 9


def test_consumer_trace_derivation():
    rec = consumer_trace.IterationRecord(
        iteration=1, n_total_rows=5,
        nonzero_grad_rows=[0, 2],
        clone_parents=[1],
        split_parents=[3],
        prune_ids=[4],
        fence_rows=[])
    cons = rec.consumer_set()
    assert cons == {0, 1, 2, 3}   # prune row 4 excluded (no SH read)
    assert 4 not in cons


def test_headroom_gates_pass():
    # craft stats that pass all gates
    stats = debt_replay.DebtStats(
        dense_row_transitions=1000, required_row_transitions=400,
        consumer_settles=200, topology_forced_settles=50,
        prune_discards=10, rows_born=20, debt_lengths=[5] * 250)
    debt_replay._finalize_percentiles(stats)
    sh = headroom_gates.SceneHeadroom(
        "room", stats, sh_adam_wall_fraction=0.2)
    assert sh.h1(), sh.as_dict()
    assert sh.h2()
    assert sh.h3()
    assert sh.h4()
    assert sh.all_pass()


def test_headroom_gates_fail_h2():
    # forced settle fraction too high
    stats = debt_replay.DebtStats(
        dense_row_transitions=1000, required_row_transitions=400,
        consumer_settles=10, topology_forced_settles=90,
        prune_discards=0, rows_born=0, debt_lengths=[5] * 100)
    debt_replay._finalize_percentiles(stats)
    sh = headroom_gates.SceneHeadroom("room", stats, sh_adam_wall_fraction=0.2)
    assert not sh.h2()
    assert not sh.all_pass()


def test_headroom_cross_scene_stop():
    # 2 of 3 scenes fail H1/H2 => stop_early
    good = debt_replay.DebtStats(
        dense_row_transitions=1000, required_row_transitions=300,
        consumer_settles=200, topology_forced_settles=20,
        prune_discards=0, rows_born=0, debt_lengths=[5] * 220)
    debt_replay._finalize_percentiles(good)
    bad = debt_replay.DebtStats(
        dense_row_transitions=1000, required_row_transitions=900,
        consumer_settles=10, topology_forced_settles=80,
        prune_discards=0, rows_born=0, debt_lengths=[1] * 90)
    debt_replay._finalize_percentiles(bad)
    scenes = {
        "room": headroom_gates.SceneHeadroom("room", good, 0.2),
        "garden": headroom_gates.SceneHeadroom("garden", bad, 0.2),
        "bicycle": headroom_gates.SceneHeadroom("bicycle", bad, 0.2),
    }
    res = headroom_gates.evaluate_headroom(scenes)
    assert res.stop_early is True
    assert res.passed is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL", len(fns), "debt_replay/gates tests passed")
