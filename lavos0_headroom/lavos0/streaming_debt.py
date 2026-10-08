"""Vectorized streaming debt replay for the REAL A100 capture (PREREG Phase B->C).

The frozen offline replay (debt_replay.replay_debt) is O(n_iters * n_alive) in
pure Python. On the real 0..15K trace (~850K SH rows x 15000 iters, ~144
topology events) that is ~1e10 inner iterations -- infeasible. This module
replays the SAME debt model with numpy vectorization over logical row ids,
processing ONE iteration at a time so the per-iteration consumer id lists can be
discarded after use (memory-bounded; no full trace is ever materialized).

It produces a debt_replay.DebtStats identical to debt_replay.replay_debt for the
same (consumers_by_iter, clone/split/prune) schedule. tests/test_streaming_debt
proves equality on randomized schedules including topology.

Debt model (STATE_MACHINE Sec.2-4), identical to debt_replay.replay_debt:
  * every iteration t: topology first (clone/split settle+birth, prune discards),
    then dense_row_transitions += |alive|, then each alive consumer settles its
    debt (debt->0) and each non-consumer accrues one deferred transition.
  * newborns start with zero m/v => zero debt; their pre-birth transitions are
    exact no-ops and are NOT counted as required settlements.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import debt_replay


def _nearest_rank_percentile(sorted_arr, pct):
    """Nearest-rank percentile matching debt_replay._percentile exactly."""
    n = sorted_arr.size
    if n == 0:
        return 0
    if n == 1:
        return int(sorted_arr[0])
    rank = pct / 100.0 * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return int(round(sorted_arr[lo] * (1 - frac) + sorted_arr[hi] * frac))


class StreamingDebtReplayer:
    """Numpy-vectorized, single-pass equivalent of debt_replay.replay_debt."""

    def __init__(self, n_initial_rows):
        self.n_initial = int(n_initial_rows)
        cap = max(16, self.n_initial * 4 + 16)
        self._cap = cap
        self.debt = np.zeros(cap, dtype=np.int64)
        self.alive = np.zeros(cap, dtype=bool)
        self.alive[:self.n_initial] = True
        self.next_id = self.n_initial
        # DebtStats accumulators (same fields as debt_replay.DebtStats)
        self.dense_row_transitions = 0
        self.required_row_transitions = 0
        self.consumer_settles = 0
        self.topology_forced_settles = 0
        self.prune_discards = 0
        self.rows_born = 0
        self._debt_len_parts: List[np.ndarray] = []

    def _ensure(self, n):
        if n <= self._cap:
            return
        newcap = self._cap
        while newcap < n:
            newcap *= 2
        d = np.zeros(newcap, dtype=np.int64)
        d[:self._cap] = self.debt
        a = np.zeros(newcap, dtype=bool)
        a[:self._cap] = self.alive
        self.debt = d
        self.alive = a
        self._cap = newcap

    def _record_debt(self, mask):
        if mask.any():
            self._debt_len_parts.append(self.debt[mask].astype(np.int64).copy())

    def replay_topology(self, clone_parents, split_parents, prune_ids):
        """Topology phase for one iteration (frozen order: before the step)."""
        for pid in (clone_parents or []):
            if self.alive[pid] and self.debt[pid] > 0:
                self.topology_forced_settles += 1
                self._debt_len_parts.append(
                    np.array([self.debt[pid]], dtype=np.int64))
                self.debt[pid] = 0
            cid = self.next_id
            self.next_id += 1
            self._ensure(cid + 1)
            self.debt[cid] = 0
            self.alive[cid] = True
            self.rows_born += 1

        for pid in (split_parents or []):
            if self.alive[pid] and self.debt[pid] > 0:
                self.topology_forced_settles += 1
                self._debt_len_parts.append(
                    np.array([self.debt[pid]], dtype=np.int64))
                self.debt[pid] = 0
            self.alive[pid] = False
            for _ in range(2):
                cid = self.next_id
                self.next_id += 1
                self._ensure(cid + 1)
                self.debt[cid] = 0
                self.alive[cid] = True
                self.rows_born += 1

        for pid in (prune_ids or []):
            if self.alive[pid]:
                if self.debt[pid] > 0:
                    self.prune_discards += 1
                self.alive[pid] = False

    def replay_step(self, consumers):
        """Dense-baseline + settle/defer phase for one iteration."""
        self.dense_row_transitions += int(self.alive.sum())
        cons = np.zeros(self._cap, dtype=bool)
        if consumers:
            ci = np.asarray(consumers, dtype=np.int64)
            if ci.size:
                self._ensure(int(ci.max()) + 1)
                cons = np.zeros(self._cap, dtype=bool)
                cons[ci] = True
        cons_alive = cons & self.alive
        settle_mask = cons_alive & (self.debt > 0)
        self.consumer_settles += int(settle_mask.sum())
        self._record_debt(settle_mask)
        self.debt[cons_alive] = 0
        noncons_alive = self.alive & ~cons
        self.debt[noncons_alive] += 1

    def finalize(self) -> debt_replay.DebtStats:
        stats = debt_replay.DebtStats()
        stats.dense_row_transitions = self.dense_row_transitions
        stats.consumer_settles = self.consumer_settles
        stats.topology_forced_settles = self.topology_forced_settles
        stats.prune_discards = self.prune_discards
        stats.rows_born = self.rows_born
        stats.required_row_transitions = (
            self.consumer_settles + self.topology_forced_settles)
        if self._debt_len_parts:
            allv = np.sort(np.concatenate(self._debt_len_parts))
        else:
            allv = np.array([], dtype=np.int64)
        if allv.size:
            stats.max_debt = int(allv[-1])
            stats.p50_debt = _nearest_rank_percentile(allv, 50)
            stats.p90_debt = _nearest_rank_percentile(allv, 90)
            stats.p99_debt = _nearest_rank_percentile(allv, 99)
        return stats
