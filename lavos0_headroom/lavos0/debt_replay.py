"""Offline logical-debt replay (PREREG Sec.4).

Given a per-iteration consumer trace (which SH rows are consumed at each
iteration) plus the topology event rows, replay a per-row logical debt counter
and derive the structural-headroom statistics that Phase C gates on.

Model (STATE_MACHINE Sec.2-4):
  * Every iteration t has a consumer set C_t of SH rows.
  * A row r NOT in C_t accrues one deferred zero-grad transition (debt += 1).
  * A row r IN C_t must be settled: all of r's pending debt (steps tau+1..t-1)
    is replayed, then r transitions at t. Debt resets to 0, and the row becomes
    the newest "materialized" row.
  * A topology event forces settlement of specific parents (clone/split copy SH
    => settle that parent; prune discards without SH read => drop debt).

From this replay we compute, per scene:
  * dense_SH_row_transitions       = sum_t N_alive(t)   (what dense A_s does)
  * required_physical_SH_row_transitions = number of settlements actually
    performed (consumer-settled + topology-forced-settled)
  * avoidable_transition_fraction  = 1 - required/dense
  * topology_forced_settles / all candidate deferred settlements
  * debt length distribution: max, p50, p90, p99
  * zero-gradient defer length distribution

This module is pure Python/CPU and is fully unit-tested.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class DebtStats:
    dense_row_transitions: int = 0
    required_row_transitions: int = 0
    consumer_settles: int = 0
    topology_forced_settles: int = 0
    prune_discards: int = 0
    rows_born: int = 0
    debt_lengths: List[int] = field(default_factory=list)
    max_debt: int = 0
    p50_debt: int = 0
    p90_debt: int = 0
    p99_debt: int = 0

    def as_dict(self):
        return {
            "dense_row_transitions": self.dense_row_transitions,
            "required_row_transitions": self.required_row_transitions,
            "avoidable_transition_fraction": self.avoidable_fraction(),
            "consumer_settles": self.consumer_settles,
            "topology_forced_settles": self.topology_forced_settles,
            "topology_forced_settle_fraction": self.forced_settle_fraction(),
            "prune_discards": self.prune_discards,
            "rows_born": self.rows_born,
            "max_debt": self.max_debt,
            "p50_debt": self.p50_debt,
            "p90_debt": self.p90_debt,
            "p99_debt": self.p99_debt,
        }

    def avoidable_fraction(self):
        if self.dense_row_transitions <= 0:
            return 0.0
        return 1.0 - (self.required_row_transitions / self.dense_row_transitions)

    def forced_settle_fraction(self):
        total_settles = self.consumer_settles + self.topology_forced_settles
        if total_settles <= 0:
            return 0.0
        return self.topology_forced_settles / total_settles


def _percentile(sorted_vals: Sequence[int], pct: float) -> int:
    """Nearest-rank percentile over a sorted (ascending) list of ints."""
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return int(sorted_vals[0])
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    # interpolate then round to nearest int debt length
    return int(round(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac))


def replay_debt(consumers_by_iter: Dict[int, Sequence[int]],
                 n_iters: int,
                 n_initial_rows: int,
                 clone_parents: Optional[Dict[int, Sequence[int]]] = None,
                 split_parents: Optional[Dict[int, Sequence[int]]] = None,
                 prune_ids: Optional[Dict[int, Sequence[int]]] = None) -> DebtStats:
    """Replay per-row debt over iterations 1..n_iters.

    Args:
      consumers_by_iter: {iteration: [row ids consumed at that iteration]}.
      n_iters: number of iterations to replay.
      n_initial_rows: number of SH rows alive at iteration 0.
      clone_parents: {iteration: [parent row ids cloned (SH copied -> settle)]}.
      split_parents: {iteration: [parent row ids split (SH read -> settle)]}.
      prune_ids:    {iteration: [row ids pruned (discarded, no settle)]}.

    Row identity is a logical id; newborns get fresh ids (they start with zero
    m/v, so their pre-birth transitions are exact no-ops and are NOT counted as
    required settlements, matching STATE_MACHINE Sec.4).
    """
    stats = DebtStats()
    clone_parents = clone_parents or {}
    split_parents = split_parents or {}
    prune_ids = prune_ids or {}

    next_id = int(n_initial_rows)
    # debt[r] = number of deferred zero-grad transitions owed by row r.
    debt: Dict[int, int] = {r: 0 for r in range(n_initial_rows)}
    alive = set(range(n_initial_rows))
    max_row_id = n_initial_rows - 1

    for t in range(1, n_iters + 1):
        # --- topology event first (frozen ordering: topology before step) ---
        for pid in clone_parents.get(t, []):
            if pid in alive and debt.get(pid, 0) > 0:
                stats.topology_forced_settles += 1
                stats.debt_lengths.append(debt[pid])
                debt[pid] = 0
            # clone: parent survives, one child born (SH copied from parent)
            cid = next_id
            next_id += 1
            debt[cid] = 0            # newborn: zero m/v => no debt
            alive.add(cid)
            max_row_id = max(max_row_id, cid)
            stats.rows_born += 1

        for pid in split_parents.get(t, []):
            if pid in alive and debt.get(pid, 0) > 0:
                stats.topology_forced_settles += 1
                stats.debt_lengths.append(debt[pid])
                debt[pid] = 0
            # split: parent removed, two children born
            alive.discard(pid)
            debt.pop(pid, None)
            for _ in range(2):
                cid = next_id
                next_id += 1
                debt[cid] = 0
                alive.add(cid)
                max_row_id = max(max_row_id, cid)
                stats.rows_born += 1

        for pid in prune_ids.get(t, []):
            if pid in alive:
                # discarded without SH/state observation: drop debt, no settle
                if debt.get(pid, 0) > 0:
                    stats.prune_discards += 1
                alive.discard(pid)
                debt.pop(pid, None)

        # --- dense baseline work this iteration: every alive row transitions ---
        stats.dense_row_transitions += len(alive)

        # --- optimizer step: settle consumers, defer the rest ---
        consumers = set(consumers_by_iter.get(t, []))
        for r in alive:
            if r in consumers:
                d = debt.get(r, 0)
                if d > 0:
                    stats.consumer_settles += 1
                    stats.debt_lengths.append(d)
                # settle all debt, then real step at t => back to zero debt
                debt[r] = 0
            else:
                # defer: accrue one zero-grad transition
                debt[r] = debt.get(r, 0) + 1

        # newborns that were not consumed this iteration start accruing debt
        # (they are alive rows of the dense baseline).

    stats.required_row_transitions = (
        stats.consumer_settles + stats.topology_forced_settles)
    _finalize_percentiles(stats)
    return stats


def _finalize_percentiles(stats: DebtStats):
    vals = sorted(stats.debt_lengths)
    if vals:
        stats.max_debt = int(vals[-1])
        stats.p50_debt = _percentile(vals, 50)
        stats.p90_debt = _percentile(vals, 90)
        stats.p99_debt = _percentile(vals, 99)
    else:
        stats.max_debt = stats.p50_debt = stats.p90_debt = stats.p99_debt = 0


def aggregate_scenes(per_scene_stats: Dict[str, DebtStats]) -> Dict[str, float]:
    """Aggregate per-scene DebtStats into the cross-scene Phase C metrics."""
    if not per_scene_stats:
        return {}
    avoidable = [s.avoidable_fraction() for s in per_scene_stats.values()]
    avoidable_sorted = sorted(avoidable)
    n = len(avoidable_sorted)
    # median
    if n % 2 == 1:
        median_avoidable = avoidable_sorted[n // 2]
    else:
        median_avoidable = 0.5 * (avoidable_sorted[n // 2 - 1] + avoidable_sorted[n // 2])
    ge_055 = sum(1 for a in avoidable if a >= 0.55)
    return {
        "median_avoidable_transition_fraction": median_avoidable,
        "min_avoidable_transition_fraction": avoidable_sorted[0],
        "max_avoidable_transition_fraction": avoidable_sorted[-1],
        "scenes_ge_0.55": ge_055,
        "scenes_ge_0.55_fraction": ge_055 / n,
    }
