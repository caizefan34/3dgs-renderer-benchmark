"""Phase D bitwise-exactness tests (PREREG Sec.6).

Proves the core LAVOS0 claim: lazy settlement of SH rows reproduces the dense
single-tensor Adam A_s lane BITWISE through clone/split/prune/compaction.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lavos0 import adam_ref                      # noqa: E402
from lavos0.topology_oracle import (              # noqa: E402
    LogicalState, TopologyOracle, TopologyEvent, Fence, exactness_verdict,
)

FEAT = 4


def _mk(seed, n=6, feat=FEAT):
    g = torch.Generator().manual_seed(seed)
    st = LogicalState.zeros(n, feat)
    st.theta = torch.randn((n, feat), generator=g)
    st.m = torch.randn((n, feat), generator=g) * 0.01
    st.v = torch.rand((n, feat), generator=g) + 0.5
    st.tau = torch.zeros(n, dtype=torch.long)
    return st


def _basic_run():
    dense = _mk(0)
    lazy = _mk(0)  # identical init
    oracle = TopologyOracle(FEAT, lr=0.002)
    grads = {1: {0: torch.randn(FEAT), 2: torch.randn(FEAT)}}
    cons = {1: [0, 2]}
    ev = TopologyEvent(iteration=2, kind="clone", parent_rows=[1],
                       child_theta_seed=dense.theta[1:2].clone())
    fences = [Fence(2, "topology"), Fence(3, "+1")]
    return oracle.run(dense, lazy, [ev], grads, cons, fences, n_iters=4)


def test_defer_and_settle_is_bitwise():
    comp = _basic_run()
    for c in comp:
        assert c["materialized_equal"], c
        assert c["step_equal"], c


def test_verdict_passes_on_exact_run():
    v = exactness_verdict(_basic_run())
    assert v["exactness_pass"] is True
    assert v["n_mismatches"] == 0


def test_split_bitwise():
    dense, lazy = _mk(1), _mk(1)
    oracle = TopologyOracle(FEAT, lr=0.002)
    grads = {1: {0: torch.randn(FEAT)}}
    cons = {1: [0]}
    ev = TopologyEvent(iteration=1, kind="split", parent_rows=[3],
                       child_theta_seed=dense.theta[3:4].repeat(2, 1).clone())
    fences = [Fence(1, "topology"), Fence(2, "+1"), Fence(3, "+8")]
    comp = oracle.run(dense, lazy, [ev], grads, cons, fences, n_iters=4)
    assert exactness_verdict(comp)["exactness_pass"] is True


def test_prune_bitwise():
    dense, lazy = _mk(2), _mk(2)
    oracle = TopologyOracle(FEAT, lr=0.002)
    grads = {1: {0: torch.randn(FEAT)}}
    cons = {1: [0]}
    ev = TopologyEvent(iteration=1, kind="prune", parent_rows=[2, 4])
    fences = [Fence(1, "topology"), Fence(2, "+1")]
    comp = oracle.run(dense, lazy, [ev], grads, cons, fences, n_iters=3)
    assert exactness_verdict(comp)["exactness_pass"] is True


def test_mixed_event_bitwise():
    dense, lazy = _mk(3), _mk(3)
    oracle = TopologyOracle(FEAT, lr=0.002)
    grads = {1: {0: torch.randn(FEAT)}, 2: {1: torch.randn(FEAT)}}
    cons = {1: [0], 2: [1]}
    events = [
        TopologyEvent(iteration=1, kind="clone", parent_rows=[2],
                      child_theta_seed=dense.theta[2:3].clone()),
        TopologyEvent(iteration=2, kind="split", parent_rows=[0],
                      child_theta_seed=dense.theta[0:1].repeat(2, 1).clone()),
        TopologyEvent(iteration=3, kind="prune", parent_rows=[5]),
    ]
    fences = [Fence(1, "topology"), Fence(2, "+1"),
              Fence(3, "topology"), Fence(4, "+8")]
    comp = oracle.run(dense, lazy, events, grads, cons, fences, n_iters=5)
    assert exactness_verdict(comp)["exactness_pass"] is True


def test_noop_event_is_identity():
    dense, lazy = _mk(4), _mk(4)
    oracle = TopologyOracle(FEAT, lr=0.002)
    grads = {1: {0: torch.randn(FEAT)}}
    cons = {1: [0]}
    ev = TopologyEvent(iteration=1, kind="noop")
    fences = [Fence(1, "topology")]
    comp = oracle.run(dense, lazy, [ev], grads, cons, fences, n_iters=2)
    assert exactness_verdict(comp)["exactness_pass"] is True


def test_real_grad_bitwise_across_fences():
    """A longer run with real grads on random rows must stay bitwise."""
    g = torch.Generator().manual_seed(11)
    n, T = 10, 12
    dense, lazy = _mk(5, n=n), _mk(5, n=n)
    oracle = TopologyOracle(FEAT, lr=0.003)
    grads_by_iter, cons_by_iter = {}, {}
    for t in range(1, T + 1):
        k = int(torch.randint(0, 4, (1,), generator=g).item())
        rows = torch.randperm(n, generator=g)[:k].tolist()
        grads_by_iter[t] = {r: torch.randn(FEAT, generator=g) for r in rows}
        cons_by_iter[t] = rows
    fences = [Fence(t, "periodic") for t in (1, 4, 8, 12)]
    comp = oracle.run(dense, lazy, [], grads_by_iter, cons_by_iter, fences,
                      n_iters=T)
    assert exactness_verdict(comp)["exactness_pass"] is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL", len(fns), "topology_oracle tests passed")
