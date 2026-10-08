"""Phase D - in-process exact topology replay oracle (PREREG Sec.6).

Purpose: prove that LAZY settlement of SH rows reproduces the DENSE single-
tensor Adam A_s lane BITWISE through clone/split/prune/compaction, before any
topology-aware CUDA is written.

The oracle uses the SAME settlement primitive (adam_ref.RowSettler, an isolated
1-row torch.optim.Adam(foreach=False)) for BOTH lanes. Because torch Adam
(foreach=False) is elementwise and cross-row independent, replaying a row's K
owed transitions reproduces the dense lane's (theta, m, v) for that row BITWISE
on any device. This removes any device-specific hand-rounded op form from the
correctness path.

Debt model (STATE_MACHINE Sec.2-4):
  * tau[r] = the last logical step through which row r has been PHYSICALLY
    materialized in this lane.
  * DENSE lane: every alive row is settled/ stepped every iteration, so after
    iteration t every row has tau == t.
  * LAZY lane: a consumer row is settled/stepped at t (tau -> t); a non-consumer
    row is DEFERRED (tau unchanged, its zero-grad debt accumulates implicitly).
    At a fence, all lazy rows are materialized up to the fence iteration so the
    bytewise comparison against the dense lane is fair.

Two lanes are compared bytewise at fences:
    DENSE : step every alive row every iteration (incl. topology newborns).
    LAZY  : defer zero-grad non-consumers; settle only exact consumers + clone/
            split parents + fence rows; same baseline topology.

Lanes operate on the SAME LogicalState bookkeeping. The only difference is WHEN
a row's owed transitions are physically applied. At a fence both are fully
materialized, so their hashes must match exactly.

If any hash differs => LAVOS0_TOPOLOGY_EXACTNESS_FAIL (no tolerance rescue).
This module is CPU/GPU agnostic and fully deterministic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from . import adam_ref
from . import hashing
from . import config


@dataclass
class LogicalState:
    theta: torch.Tensor          # [N, F]
    m: torch.Tensor              # [N, F]
    v: torch.Tensor              # [N, F]
    tau: torch.Tensor            # [N] int64 last materialized logical step
    alive: torch.Tensor          # [N] bool
    n_rows: int = 0
    feat: int = 0

    @staticmethod
    def zeros(n, feat, device="cpu", dtype=torch.float32):
        return LogicalState(
            theta=torch.zeros((n, feat), dtype=dtype, device=device),
            m=torch.zeros((n, feat), dtype=dtype, device=device),
            v=torch.zeros((n, feat), dtype=dtype, device=device),
            tau=torch.zeros((n,), dtype=torch.long, device=device),
            alive=torch.ones((n,), dtype=torch.bool, device=device),
            n_rows=n, feat=feat)

    def materialized_hash(self):
        """SHA256 over alive rows' theta/m/v (bytewise)."""
        idx = torch.where(self.alive)[0]
        return hashing.hash_tensor_list([
            self.theta[idx].contiguous(),
            self.m[idx].contiguous(),
            self.v[idx].contiguous(),
        ])

    def step_hash(self):
        idx = torch.where(self.alive)[0]
        return hashing.hash_tensor_list([
            self.tau[idx].contiguous(),
            torch.tensor([int(self.alive.sum())], dtype=torch.long),
        ])

    def child_count_hash(self):
        idx = torch.where(self.alive)[0]
        return hashing.hash_tensor_list([idx.contiguous()])


def _append_rows(st: LogicalState, theta, m, v):
    st.theta = torch.cat([st.theta, theta], dim=0)
    st.m = torch.cat([st.m, m], dim=0)
    st.v = torch.cat([st.v, v], dim=0)
    st.tau = torch.cat([st.tau, torch.zeros(theta.shape[0], dtype=torch.long,
                                             device=st.theta.device)], dim=0)
    st.alive = torch.cat([st.alive, torch.ones(theta.shape[0], dtype=torch.bool,
                                                device=st.theta.device)], dim=0)
    st.n_rows = st.theta.shape[0]


def _drop_rows(st: LogicalState, mask_keep: torch.Tensor):
    st.theta = st.theta[mask_keep].contiguous()
    st.m = st.m[mask_keep].contiguous()
    st.v = st.v[mask_keep].contiguous()
    st.tau = st.tau[mask_keep].contiguous()
    st.alive = st.alive[mask_keep].contiguous()
    st.n_rows = st.theta.shape[0]


@dataclass
class TopologyEvent:
    iteration: int
    kind: str                       # "clone" | "split" | "prune" | "mixed" | "noop"
    parent_rows: List[int] = field(default_factory=list)
    child_theta_seed: Optional[torch.Tensor] = None  # [K,F] for children
    global_step_at_event: int = 0


@dataclass
class Fence:
    iteration: int
    kind: str                       # "topology" | "+1" | "+8" | "eval" | "final"


class TopologyOracle:
    """Runs the DENSE and LAZY lanes over an identical event+gradient schedule
    and compares them bytewise at each fence."""

    def __init__(self, feat, lr, betas=config.ADAM_BETAS, eps=config.ADAM_EPS,
                 device="cpu"):
        self.feat = feat
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.device = device

    # -- settlement helpers (use the shared RowSettler primitive) ------------
    def _settle_zero(self, st, row, target_step):
        tau = int(st.tau[row].item())
        if target_step <= tau:
            return
        th, m, v, nt = adam_ref.settle_row_zero(
            st.theta[row], st.m[row], st.v[row], tau, target_step, self.feat,
            self.lr, betas=self.betas, eps=self.eps, device=self.device)
        st.theta[row] = th
        st.m[row] = m
        st.v[row] = v
        st.tau[row] = nt

    def _step_real(self, st, row, target_step, grad):
        tau = int(st.tau[row].item())
        th, m, v, nt = adam_ref.settle_row_real(
            st.theta[row], st.m[row], st.v[row], tau, target_step, grad,
            self.feat, self.lr, betas=self.betas, eps=self.eps,
            device=self.device)
        st.theta[row] = th
        st.m[row] = m
        st.v[row] = v
        st.tau[row] = nt

    # -- topology ------------------------------------------------------------
    def _apply_topology(self, st: LogicalState, ev: TopologyEvent, t: int,
                        settle: bool):
        if ev.kind == "noop":
            return
        rows = [r for r in ev.parent_rows
                if 0 <= r < st.n_rows and bool(st.alive[r])]
        if ev.kind in ("clone", "split", "mixed"):
            # parents whose SH is read => settle them first (lazy lane only;
            # dense lane already has tau == t-1 so this is a no-op there).
            if settle:
                for r in rows:
                    self._settle_zero(st, r, t - 1)
            if ev.child_theta_seed is not None:
                child_theta = ev.child_theta_seed.clone().to(self.device)
                child_m = torch.zeros_like(child_theta)
                child_v = torch.zeros_like(child_theta)
                _append_rows(st, child_theta, child_m, child_v)
            if ev.kind == "split":
                keep = torch.ones(st.n_rows, dtype=torch.bool,
                                  device=st.theta.device)
                for r in rows:
                    keep[r] = False
                _drop_rows(st, keep)
        elif ev.kind == "prune":
            keep = torch.ones(st.n_rows, dtype=torch.bool,
                              device=st.theta.device)
            for r in rows:
                keep[r] = False
            _drop_rows(st, keep)

    # -- per-iteration stepping ---------------------------------------------
    def _step_lane(self, st: LogicalState, t: int, grads, consumers,
                   settle_all):
        if settle_all:
            for r in range(st.n_rows):
                if not bool(st.alive[r]):
                    continue
                g = grads.get(r)
                if g is None:
                    self._settle_zero(st, r, t)
                else:
                    self._step_real(st, r, t, g)
        else:
            for r in range(st.n_rows):
                if not bool(st.alive[r]):
                    continue
                g = grads.get(r)
                if g is not None:
                    self._step_real(st, r, t, g)
                elif r in consumers:
                    self._settle_zero(st, r, t)
                else:
                    pass  # defer: tau unchanged, debt accumulates implicitly

    def _materialize_upto(self, st, t):
        """Settle every alive row's owed zero-grad transitions up to step t.

        Used at fences so the lazy lane is fully materialized before a bytewise
        comparison against the dense lane (which steps every row every iter).
        Idempotent for rows already materialized through t.
        """
        for r in range(st.n_rows):
            if not bool(st.alive[r]):
                continue
            if int(st.tau[r].item()) < t:
                self._settle_zero(st, r, t)

    def run(self, dense: LogicalState, lazy: LogicalState,
            events: Sequence[TopologyEvent],
            grads_by_iter: Dict[int, Dict[int, torch.Tensor]],
            consumers_by_iter: Dict[int, Sequence[int]],
            fences: Sequence[Fence],
            n_iters: int):
        fence_by_iter: Dict[int, List[Fence]] = {}
        for f in fences:
            fence_by_iter.setdefault(f.iteration, []).append(f)
        event_by_iter: Dict[int, List[TopologyEvent]] = {}
        for e in events:
            event_by_iter.setdefault(e.iteration, []).append(e)

        comparisons = []
        for t in range(1, n_iters + 1):
            for ev in event_by_iter.get(t, []):
                self._apply_topology(dense, ev, t, settle=False)
                self._apply_topology(lazy, ev, t, settle=True)

            grads = grads_by_iter.get(t, {})
            consumers = set(consumers_by_iter.get(t, []))
            self._step_lane(dense, t, grads, consumers, settle_all=True)
            self._step_lane(lazy, t, grads, consumers, settle_all=False)

            for f in fence_by_iter.get(t, []):
                self._materialize_upto(lazy, t)
                self._materialize_upto(dense, t)
                comparisons.append(self._compare(dense, lazy, t, f.kind))
        return comparisons

    def _compare(self, dense, lazy, t, kind):
        dh = dense.materialized_hash()
        lh = lazy.materialized_hash()
        ds = dense.step_hash()
        ls = lazy.step_hash()
        dc = dense.child_count_hash()
        lc = lazy.child_count_hash()
        return {
            "iteration": t,
            "fence": kind,
            "dense_materialized_hash": dh,
            "lazy_materialized_hash": lh,
            "dense_step_hash": ds,
            "lazy_step_hash": ls,
            "dense_child_hash": dc,
            "lazy_child_hash": lc,
            "n_dense": int(dense.alive.sum()),
            "n_lazy": int(lazy.alive.sum()),
            "materialized_equal": dh == lh,
            "step_equal": ds == ls,
            "child_equal": dc == lc,
        }


def exactness_verdict(comparisons: Sequence[dict]) -> dict:
    """Reduce Phase D comparisons to a verdict payload (PREREG Sec.6)."""
    ok = bool(comparisons) and all(
        c["materialized_equal"] and c["step_equal"] and c["child_equal"]
        and c["n_dense"] == c["n_lazy"]
        for c in comparisons)
    mismatches = [c for c in comparisons
                  if not (c["materialized_equal"] and c["step_equal"]
                          and c["child_equal"] and c["n_dense"] == c["n_lazy"])]
    return {
        "exactness_pass": ok,
        "n_fences": len(comparisons),
        "n_mismatches": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
    }
