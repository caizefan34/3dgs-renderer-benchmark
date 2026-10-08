"""Phase B - consumer trace / headroom oracle (PREREG Sec.4).

Runs ONE authoritative dense A_s 0..15K trace per scene and, per iteration,
records device-side where practical:
    N_total_SH_rows, N_nonzero_SH_grad_rows, N_zero_SH_grad_rows,
    clone/split parent IDs, prune IDs, new child count, rows remapped/compacted,
    topology-event wall, dense SH Adam wall, total optimizer wall, total iter wall.

From these it derives the MINIMAL exact-consumer set (PREREG Sec.4):
    C_t = nonzero_SH_grad_rows
          UNION SH rows whose value/state is read by the frozen topology op
          UNION explicit fence rows

Pruned rows discarded WITHOUT SH/state observation are NOT consumers.

This module defines the data schema and the consumer-set derivation. The actual
per-iteration capture is an A100 driver concern (see drivers/); the pure
derivation logic (which rows are consumers, how debt stats roll up) is CPU and
unit-tested here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set

from . import debt_replay


@dataclass
class IterationRecord:
    iteration: int
    n_total_rows: int
    nonzero_grad_rows: List[int] = field(default_factory=list)
    clone_parents: List[int] = field(default_factory=list)
    split_parents: List[int] = field(default_factory=list)
    prune_ids: List[int] = field(default_factory=list)
    new_child_count: int = 0
    rows_compacted: int = 0
    fence_rows: List[int] = field(default_factory=list)
    # walls (ms); None on the CPU/schema path, filled by the A100 driver
    topology_wall_ms: float = None
    dense_sh_adam_wall_ms: float = None
    total_optimizer_wall_ms: float = None
    total_iter_wall_ms: float = None

    def consumer_set(self) -> Set[int]:
        """Minimal exact-consumer set C_t for this iteration.

        nonzero_grad_rows  -> real (nonzero) SH gradient consumers
        clone/split parents -> topology reads/copies SH state (settle)
        fence_rows         -> eval/checkpoint/debug reads (fence: settle all)
        prune_ids          -> NOT consumers unless their SH is observed first
                              (prune discards without read => excluded here).
        """
        c: Set[int] = set(self.nonzero_grad_rows)
        c.update(self.clone_parents)
        c.update(self.split_parents)
        c.update(self.fence_rows)
        return c

    def zero_grad_rows(self, all_rows: Sequence[int]) -> List[int]:
        cons = self.consumer_set()
        return [r for r in all_rows if r not in cons]

    def as_dict(self):
        return {
            "iteration": self.iteration,
            "n_total_rows": self.n_total_rows,
            "n_nonzero_grad_rows": len(self.nonzero_grad_rows),
            "n_zero_grad_rows": self.n_total_rows - len(self.nonzero_grad_rows),
            "clone_parents": list(self.clone_parents),
            "split_parents": list(self.split_parents),
            "prune_ids": list(self.prune_ids),
            "new_child_count": self.new_child_count,
            "rows_compacted": self.rows_compacted,
            "fence_rows": list(self.fence_rows),
            "consumer_count": len(self.consumer_set()),
            "topology_wall_ms": self.topology_wall_ms,
            "dense_sh_adam_wall_ms": self.dense_sh_adam_wall_ms,
            "total_optimizer_wall_ms": self.total_optimizer_wall_ms,
            "total_iter_wall_ms": self.total_iter_wall_ms,
        }


def build_consumer_maps(records: Sequence[IterationRecord]):
    """Turn IterationRecords into the {iter: [consumer rows]} map for debt replay."""
    consumers_by_iter: Dict[int, List[int]] = {}
    clone_by_iter: Dict[int, List[int]] = {}
    split_by_iter: Dict[int, List[int]] = {}
    prune_by_iter: Dict[int, List[int]] = {}
    for rec in records:
        consumers_by_iter[rec.iteration] = sorted(rec.consumer_set())
        clone_by_iter[rec.iteration] = list(rec.clone_parents)
        split_by_iter[rec.iteration] = list(rec.split_parents)
        prune_by_iter[rec.iteration] = list(rec.prune_ids)
    return consumers_by_iter, clone_by_iter, split_by_iter, prune_by_iter


def run_headroom_oracle(records: Sequence[IterationRecord],
                        n_initial_rows: int,
                        n_iters: int) -> debt_replay.DebtStats:
    """Full Phase B->C roll-up: consumer maps + offline debt replay."""
    consumers_by_iter, clone_by_iter, split_by_iter, prune_by_iter = \
        build_consumer_maps(records)
    return debt_replay.replay_debt(
        consumers_by_iter=consumers_by_iter,
        n_iters=n_iters,
        n_initial_rows=n_initial_rows,
        clone_parents=clone_by_iter,
        split_parents=split_by_iter,
        prune_ids=prune_by_iter,
    )


def summarize_trace(records: Sequence[IterationRecord]) -> Dict[str, float]:
    """Aggregate wall + count statistics over a trace (for LAVOS0_CONSUMER_TRACE)."""
    if not records:
        return {"n_iterations": 0}
    topo = [r.topology_wall_ms for r in records if r.topology_wall_ms is not None]
    adam = [r.dense_sh_adam_wall_ms for r in records if r.dense_sh_adam_wall_ms is not None]
    opt = [r.total_optimizer_wall_ms for r in records if r.total_optimizer_wall_ms is not None]
    itw = [r.total_iter_wall_ms for r in records if r.total_iter_wall_ms is not None]
    nz = [len(r.nonzero_grad_rows) for r in records]
    nt = [r.n_total_rows for r in records]

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    def _sum(xs):
        return sum(xs) if xs else None

    return {
        "n_iterations": len(records),
        "n_total_rows_mean": _mean(nt),
        "n_nonzero_grad_rows_mean": _mean(nz),
        "nonzero_grad_fraction_mean": (sum(nz) / sum(nt)) if sum(nt) else None,
        "topology_wall_ms_total": _sum(topo),
        "dense_sh_adam_wall_ms_total": _sum(adam),
        "total_optimizer_wall_ms_total": _sum(opt),
        "total_iter_wall_ms_total": _sum(itw),
        "sh_adam_wall_fraction": (_sum(adam) / _sum(itw)) if (_sum(adam) and _sum(itw)) else None,
    }
