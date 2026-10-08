"""Frozen configuration for LAVOS0 (from LAVOS0_PREREG.md + LAVOS0_SOURCE_LOCK.json).

These constants are frozen by the preregistration. They are deliberately kept
in one place so that every phase module reads identical values and the
no-rescue rules (Sec.11) are enforced structurally rather than by convention.
"""

# ---------------------------------------------------------------------------
# Scenes / split / seed  (SOURCE_LOCK.data_protocol + PREREG Sec.2)
# ---------------------------------------------------------------------------
SCENES = ("room", "garden", "bicycle")
SEED = 42
# corrected true-holdout: original camera index i % 8 == 0 excluded from training
HOLDOUT_MOD = 8
HOLDOUT_REMAINDER = 0

# Phase scope: P0 measures the 0..15000 dynamic-topology phase only.
PHASE_START_ITER = 0
PHASE_END_ITER = 15000            # inclusive upper bound of the measured phase
FULL_RUN_ITER = 30000             # context for the full-run incremental gain

# Parameter scope: SH DC + SH rest ONLY (matches VOS0/VOS1).
SH_SCOPE = ("shs_dc", "shs_rest")

# ---------------------------------------------------------------------------
# Optimizer semantic target  (PREREG Sec.2): dense single-tensor Adam A_s
# ---------------------------------------------------------------------------
# torch.optim.Adam ctor parameters used by the frozen C0 trainer.
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-15
ADAM_WEIGHT_DECAY = 0.0
ADAM_AMSGRAD = False
ADAM_FOREACH = False              # A_s == single-tensor foreach=False path

# ---------------------------------------------------------------------------
# Structural headroom gates (PREREG Sec.5) -- per scene
# ---------------------------------------------------------------------------
GATE_H1_AVOIDABLE_MEDIAN = 0.40   # avoidable_transition_fraction median >= 0.40
GATE_H2_FORCED_SETTLE = 0.25      # topology_forced_settle_fraction <= 0.25
GATE_H3_P90_DEBT = 3              # p90 debt length >= 3 iterations
GATE_H4_SH_WALL_FRACTION = 0.10   # SH Adam wall fraction >= 0.10 of iter wall
# across scenes: >= 2/3 must have avoidable_transition_fraction >= 0.55
CROSS_SCENE_AVOIDABLE = 0.55
CROSS_SCENE_MIN_FRACTION = 2.0 / 3.0

# ---------------------------------------------------------------------------
# Performance gates (PREREG Sec.8)
# ---------------------------------------------------------------------------
PERF_MIN_SPEEDUP_EVERY = 1.35      # early SH optimizer speedup >= 1.35x/scene
PERF_GEOMEAN_SPEEDUP = 1.50        # geometric mean >= 1.50x
PERF_MIN_FULL30K_REDUCTION = 0.02  # conservative incremental full-30K >= 2% in >=2/3
PERF_MAX_REGRESSION = -0.005       # no scene predicted regression > 0.5%
PERF_MAX_OVERHEAD_FRACTION = 0.25  # topology/fence overhead <= 25% of gross savings
STRONG_MIN_SPEEDUP_EVERY = 1.75
STRONG_MIN_FULL30K_REDUCTION = 0.03
STRONG_MAX_OVERHEAD_FRACTION = 0.15

# ---------------------------------------------------------------------------
# Microbench checkpoints (PREREG Sec.7.E) within 0..15K
# ---------------------------------------------------------------------------
MICROBENCH_ITERS = (1000, 5000, 10000, 14500)

# ---------------------------------------------------------------------------
# Output artifact names (PREREG Sec.12). The frozen analyzer reads SUMMARY and
# writes VERDICT by replacing the trailing "SUMMARY" with "VERDICT".
# ---------------------------------------------------------------------------
ARTIFACTS = (
    "LAVOS0_ENVIRONMENT.json",
    "LAVOS0_SOURCE_IDENTITY.json",
    "LAVOS0_EVENT_ORDER.json",
    "LAVOS0_CONSUMER_TRACE.json",
    "LAVOS0_HEADROOM.json",
    "LAVOS0_TOPOLOGY_ORACLE.json",
    "LAVOS0_PERF_ORACLE.json",
    "LAVOS0_SUMMARY.json",
    "LAVOS0_VERDICT.json",
    "LAVOS0_REPORT.md",
    "MANIFEST.sha256",
)

# Verdict labels (PREREG Sec.9 mechanical precedence, most severe first).
VERDICT_PRECEDENCE = (
    "LAVOS0_SOURCE_IDENTITY_FAIL",
    "LAVOS0_EVENT_SEMANTICS_UNRESOLVED",
    "LAVOS0_NO_STRUCTURAL_HEADROOM",
    "LAVOS0_TOPOLOGY_EXACTNESS_FAIL",
    "LAVOS0_PERFORMANCE_HEADROOM_FAIL",
    "LAVOS0_HEADROOM_PROMOTE",
    "LAVOS0_STRONG_HEADROOM_PROMOTE",
)


def holdout_camera_indices(num_cameras):
    """Return the corrected true-holdout camera indices (i % 8 == 0)."""
    return [i for i in range(num_cameras)
            if i % HOLDOUT_MOD == HOLDOUT_REMAINDER]


def is_holdout_camera(index):
    """True if camera `index` is excluded from training (i % 8 == 0)."""
    return index % HOLDOUT_MOD == HOLDOUT_REMAINDER
