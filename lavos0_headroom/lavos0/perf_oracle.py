"""Phase E - performance oracle, not a full implementation (PREREG Sec.7-8).

Do NOT build a production topology-aware optimizer here. This module computes:
  * the optimizer-stage predicted speedup from recorded consumer fractions;
  * clone/split forced-settlement batches and compaction/remap metadata cost;
  * the conservative Amdahl / traffic estimate of 0..15K wall saved;
  * the predicted incremental full-30K wall reduction.

Two independent estimates are produced and the LOWER (conservative) is used, as
required by PREREG Sec.8:
  E_amdahl   : traffic / Amdahl bound from the measured consumer fractions.
  E_microbench : from the microbenchmark primitives (A100 driver supplies the
                 raw per-op timings; this module folds them into a speedup).

Reported gates:
  * predicted early SH optimizer speedup >= 1.35x every scene, geomean >= 1.50x
  * conservative incremental full-30K wall reduction >= 2% in >= 2/3 scenes
  * no scene predicted regression > 0.5%
  * topology/fence overhead <= 25% of gross early optimizer savings
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config


def geometric_mean(xs: List[float]) -> float:
    xs = [x for x in xs if x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def amdahl_optimizer_speedup(consumer_fraction: float) -> float:
    """Amdahl speedup of the SH optimizer stage when only `consumer_fraction`
    of rows need real work and the rest are deferred to rare settlements.

    Dense stage time ~ 1.0 (all rows). Lazy stage time ~ consumer_fraction
    (real work) + settlement_overhead. We model settlement as amortised into
    the consumer fraction (deferred rows settle only when consumed), so the
    ideal speedup is 1 / max(consumer_fraction, eps). This is the traffic bound.
    """
    cf = max(min(consumer_fraction, 1.0), 1e-6)
    return 1.0 / cf


@dataclass
class ScenePerf:
    scene: str
    consumer_fraction: float                 # mean fraction of SH rows consumed
    sh_adam_wall_fraction: float              # SH Adam wall / iter wall (0..15K)
    t_pre_wall: Optional[float] = None        # measured 0..15K wall (ms)
    t_post_wall: Optional[float] = None       # authoritative 15K..30K estimate
    existing_post15k_saving: float = 0.0      # ms already saved post-15K
    microbench_speedup: Optional[float] = None
    forced_settle_overhead: float = 0.0       # extra topology/fence work (ms/iter)
    dense_sh_adam_ms_per_iter: Optional[float] = None
    lazy_sh_adam_ms_per_iter: Optional[float] = None

    def optimizer_stage_speedup(self) -> float:
        """Predicted early SH optimizer speedup = lower of Amdahl and microbench."""
        e_amdahl = amdahl_optimizer_speedup(self.consumer_fraction)
        if self.microbench_speedup is not None:
            return min(e_amdahl, self.microbench_speedup)
        return e_amdahl

    def gross_saving_ms_per_iter(self) -> float:
        """Gross 0..15K SH-optimizer ms saved per iteration (before overhead)."""
        if self.dense_sh_adam_ms_per_iter is None:
            return 0.0
        speedup = self.optimizer_stage_speedup()
        if speedup <= 0:
            return 0.0
        return self.dense_sh_adam_ms_per_iter * (1.0 - 1.0 / speedup)

    def net_saving_ms_per_iter(self) -> float:
        return max(self.gross_saving_ms_per_iter() - self.forced_settle_overhead, 0.0)

    def overhead_fraction_of_gross(self) -> float:
        gross = self.gross_saving_ms_per_iter()
        if gross <= 0:
            return 0.0
        return self.forced_settle_overhead / gross

    def delta_pre_wall(self) -> float:
        """Predicted absolute LAVOS saving over 0..15K (ms)."""
        if self.t_pre_wall is None:
            return 0.0
        # net per-iter saving scaled by the dense SH Adam fraction of iter wall
        dense_sh_total = self.t_pre_wall * self.sh_adam_wall_fraction
        if dense_sh_total <= 0:
            return 0.0
        speedup = self.optimizer_stage_speedup()
        saved = dense_sh_total * (1.0 - 1.0 / speedup)
        # subtract overhead expressed as a fraction of gross
        saved = saved * (1.0 - self.overhead_fraction_of_gross())
        return max(saved, 0.0)

    def incremental_full30k_reduction_fraction(self) -> float:
        """G_incremental = Delta_pre / (T_pre + T_post - existing_post15k_saving).

        Uses the LOWER of the Amdahl and microbench estimates (already folded
        into optimizer_stage_speedup). Returns a signed fraction; a negative
        value is a predicted regression.
        """
        delta_pre = self.delta_pre_wall()
        denom = None
        if self.t_pre_wall is not None and self.t_post_wall is not None:
            denom = self.t_pre_wall + self.t_post_wall - self.existing_post15k_saving
        if not denom or denom <= 0 or delta_pre <= 0:
            return 0.0
        return delta_pre / denom

    def all_pass(self):
        return (self.optimizer_stage_speedup() >= config.PERF_MIN_SPEEDUP_EVERY
                and self.incremental_full30k_reduction_fraction()
                >= config.PERF_MIN_FULL30K_REDUCTION
                and self.incremental_full30k_reduction_fraction()
                > config.PERF_MAX_REGRESSION
                and self.overhead_fraction_of_gross()
                <= config.PERF_MAX_OVERHEAD_FRACTION)

    def is_strong(self):
        return (self.optimizer_stage_speedup() >= config.STRONG_MIN_SPEEDUP_EVERY
                and self.incremental_full30k_reduction_fraction()
                >= config.STRONG_MIN_FULL30K_REDUCTION
                and self.overhead_fraction_of_gross()
                <= config.STRONG_MAX_OVERHEAD_FRACTION)

    def as_dict(self):
        return {
            "scene": self.scene,
            "consumer_fraction": self.consumer_fraction,
            "sh_adam_wall_fraction": self.sh_adam_wall_fraction,
            "optimizer_stage_speedup": self.optimizer_stage_speedup(),
            "microbench_speedup": self.microbench_speedup,
            "gross_saving_ms_per_iter": self.gross_saving_ms_per_iter(),
            "net_saving_ms_per_iter": self.net_saving_ms_per_iter(),
            "forced_settle_overhead_ms_per_iter": self.forced_settle_overhead,
            "overhead_fraction_of_gross": self.overhead_fraction_of_gross(),
            "delta_pre_wall_ms": self.delta_pre_wall(),
            "incremental_full30k_reduction_fraction":
                self.incremental_full30k_reduction_fraction(),
            "pass": self.all_pass(),
            "strong": self.is_strong(),
        }


def evaluate_performance(per_scene: Dict[str, ScenePerf]) -> dict:
    """Aggregate Phase E across scenes + the cross-scene gates."""
    per = {name: p.as_dict() for name, p in per_scene.items()}
    speedups = [p.optimizer_stage_speedup() for p in per_scene.values()]
    geomean_speedup = geometric_mean(speedups) if speedups else 0.0
    reductions = [p.incremental_full30k_reduction_fraction()
                  for p in per_scene.values()]
    reductions_ok = sum(1 for r in reductions
                        if r >= config.PERF_MIN_FULL30K_REDUCTION)
    n = max(len(per_scene), 1)
    regressions = [r for r in reductions if r < config.PERF_MAX_REGRESSION]
    overhead_ok = all(
        p.overhead_fraction_of_gross() <= config.PERF_MAX_OVERHEAD_FRACTION
        for p in per_scene.values())
    every_speedup = all(
        s >= config.PERF_MIN_SPEEDUP_EVERY for s in speedups)

    passed = bool(every_speedup and geomean_speedup >= config.PERF_GEOMEAN_SPEEDUP
                  and reductions_ok / n >= config.CROSS_SCENE_MIN_FRACTION
                  and not regressions and overhead_ok)
    strong = bool(passed and every_speedup
                  and all(p.is_strong() for p in per_scene.values())
                  and geomean_speedup >= config.STRONG_MIN_SPEEDUP_EVERY)
    return {
        "per_scene": per,
        "geomean_optimizer_speedup": geomean_speedup,
        "scenes_full30k_reduction_ge_2pct": reductions_ok,
        "scenes_full30k_reduction_fraction": reductions_ok / n,
        "n_regressions_gt_0.5pct": len(regressions),
        "overhead_within_gate": overhead_ok,
        "every_scene_speedup_ge_1.35": every_speedup,
        "performance_headroom_pass": passed,
        "strong_headroom_pass": strong,
    }
