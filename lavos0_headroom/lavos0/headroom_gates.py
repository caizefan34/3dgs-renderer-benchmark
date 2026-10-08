"""Phase C - structural headroom gates (PREREG Sec.5).

Per scene require:
    H1. avoidable_transition_fraction median >= 0.40
    H2. topology_forced_settle_fraction <= 0.25
    H3. p90 debt length >= 3 iterations
    H4. SH optimizer wall fraction over 0..15K >= 0.10 of iteration wall
Across scenes:
    at least 2/3 scenes must have avoidable_transition_fraction >= 0.55.

If H1/H2 fail in >= 2 scenes:
    LAVOS0_NO_STRUCTURAL_HEADROOM  -> STOP before topology-aware CUDA.

These are MECHANISM gates, not quality gates.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from . import config
from . import debt_replay


@dataclass
class SceneHeadroom:
    scene: str
    debt_stats: debt_replay.DebtStats
    sh_adam_wall_fraction: Optional[float] = None

    def h1(self):
        return self.debt_stats.avoidable_fraction() >= config.GATE_H1_AVOIDABLE_MEDIAN

    def h2(self):
        return self.debt_stats.forced_settle_fraction() <= config.GATE_H2_FORCED_SETTLE

    def h3(self):
        return self.debt_stats.p90_debt >= config.GATE_H3_P90_DEBT

    def h4(self):
        if self.sh_adam_wall_fraction is None:
            return False
        return self.sh_adam_wall_fraction >= config.GATE_H4_SH_WALL_FRACTION

    def avoidable_ge_055(self):
        return self.debt_stats.avoidable_fraction() >= config.CROSS_SCENE_AVOIDABLE

    def all_pass(self):
        return self.h1() and self.h2() and self.h3() and self.h4()

    def as_dict(self):
        return {
            "scene": self.scene,
            "avoidable_transition_fraction":
                self.debt_stats.avoidable_fraction(),
            "topology_forced_settle_fraction":
                self.debt_stats.forced_settle_fraction(),
            "p90_debt": self.debt_stats.p90_debt,
            "max_debt": self.debt_stats.max_debt,
            "sh_adam_wall_fraction": self.sh_adam_wall_fraction,
            "H1_avoidable_ge_0.40": self.h1(),
            "H2_forced_settle_le_0.25": self.h2(),
            "H3_p90_debt_ge_3": self.h3(),
            "H4_sh_wall_ge_0.10": self.h4(),
            "avoidable_ge_0.55": self.avoidable_ge_055(),
            "all_pass": self.all_pass(),
        }


@dataclass
class HeadroomGateResult:
    per_scene: Dict[str, dict]
    aggregate: dict
    passed: bool
    stop_early: bool

    def as_dict(self):
        return {
            "per_scene": self.per_scene,
            "aggregate": self.aggregate,
            "passed": self.passed,
            "stop_early": self.stop_early,
        }


def evaluate_headroom(scenes: Dict[str, SceneHeadroom]) -> HeadroomGateResult:
    """Evaluate all Phase C gates across scenes.

    `scenes` maps scene name -> SceneHeadroom (with measured DebtStats and the
    measured SH-Adam wall fraction; H4 may be None on the schema path, in which
    case H4 fails closed).
    """
    per_scene = {name: sh.as_dict() for name, sh in scenes.items()}
    agg = debt_replay.aggregate_scenes(
        {name: sh.debt_stats for name, sh in scenes.items()})

    per_scene_all_pass = all(sh.all_pass() for sh in scenes.values())
    cross_ok = agg.get("scenes_ge_0.55_fraction", 0.0) >= config.CROSS_SCENE_MIN_FRACTION

    # H1/H2 fail in >= 2 scenes => NO_STRUCTURAL_HEADROOM stop condition.
    h1h2_fail = sum(
        1 for sh in scenes.values()
        if (not sh.h1()) or (not sh.h2()))
    stop_early = h1h2_fail >= 2

    passed = bool(per_scene_all_pass and cross_ok)
    return HeadroomGateResult(
        per_scene=per_scene,
        aggregate=agg,
        passed=passed,
        stop_early=stop_early,
    )
