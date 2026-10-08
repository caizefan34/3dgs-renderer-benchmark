"""LAVOS0 - Lineage-Aware Exact VOS P0 experiment harness.

This package implements the frozen LAVOS0 preregistration
(LAVOS0_PREREG.md, LAVOS0_STATE_MACHINE.md) as a fail-closed experiment that
decides whether a full dynamic-topology exact VOS (LAVOS1) is worth
implementing.

Design constraints (from the prereg / DSH handoff):
  * SH DC + SH rest ONLY (matches VOS0/VOS1); no opacity/xyz/scale/quat work.
  * Optimizer semantic target: dense PyTorch single-tensor Adam A_s
    (torch.optim.Adam, foreach=False).
  * Phase scope: iterations 0..15000, scenes {room, garden, bicycle}, seed 42.
  * VOS1 and ReproTopo0 are FROZEN and MUST NOT be modified. This harness only
    imports their deterministic topology/state-machine primitives.
  * No scientific verdict is fabricated: GPU measurement phases require the
    EPIC-05 A100 and are provided as drivers/runbook, not run here.

The deterministic logic (event ordering, consumer/debt replay, structural
gates, exact-replay oracle, Amdahl estimate) is CPU-unit-tested in tests/.
"""

__all__ = [
    "config", "hashing", "adam_ref", "event_order", "consumer_trace",
    "debt_replay", "headroom_gates", "topology_oracle", "perf_oracle",
    "summary", "verdict",
]

__version__ = "1.0.0"
