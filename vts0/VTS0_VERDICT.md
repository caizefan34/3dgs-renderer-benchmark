# VTS0 verdict — Visibility-Topology Scheduler feasibility audit
Date: 2026-10-01
Execution: EPIC-05 A100 (8x A100-SXM4-80GB, torch 2.9.1+cu128, seed 42)
Method: zero-diff visibility + topology trace on frozen C0 (VTS0_TRACE_DRIVER),
4000-iteration dynamic-phase window (iter 1..4000, densify active 501..3999),
35 topology events captured per scene. NO modification to VOS/ReproTopo. No
threshold relaxation.

question: |
  Does topology-aware dependency scheduling provide ADDITIONAL safe optimizer
  skipping beyond visibility-based VOS?

verdict: VTS0_DROP
gate: extra_safe_skip_headroom < 10% -> CLOSE DIRECTION

measured_ratios:
  room:
    visibility_skip_ratio: 0.6211
    topology_forced_update_ratio: 0.0479
    lineage_dependency_ratio: 0.0645
    clone_split_prune_settlement_ratio: 0.0612
    extra_safe_skip_headroom_ratio: 0.0
  bicycle:
    visibility_skip_ratio: 0.7458
    topology_forced_update_ratio: 0.0942
    lineage_dependency_ratio: 0.1435
    clone_split_prune_settlement_ratio: 0.1007
    extra_safe_skip_headroom_ratio: 0.0

structural_argument: |
  1. VOS already skips ALL invisible rows (visibility_skip_ratio 0.62-0.75).
  2. A visible row has a real nonzero gradient => its Adam transition is real;
     skipping it changes the training trajectory (not exact). Topology cannot
     make a visible row's gradient zero.
  3. Topology events (clone/split/prune) only ADD forced reads/copies/settlements
     (they consume optimizer state), they never enable skipping a visible row's
     real step. Hence topology_forced/lineage/settlement ratios are COSTS, not
     savings.
  4. In the VOS active region (iter 15001..30000) topology is FROZEN
     (densify_until_iter=15000 gates clone/split/prune AND opacity reset), so all
     three topology ratios are exactly 0 there and extra headroom from topology
     is 0 by construction.
  Therefore extra_safe_skip_headroom = 0 in the VOS region, and in the dynamic
  region topology only REDUCES headroom (forced settlements). 0 < 10% => CLOSE.

decisive_metric: extra_safe_skip_headroom_ratio = 0.0 (both scenes)
why_killed: |
  Topology-awareness provides zero additional safe optimizer skipping beyond
  visibility. The only skip signal VOS can safely exploit is invisibility
  (zero-grad => exact deferred replay). Topology events force settlements, they
  do not create skip opportunities. This is a structural property of dense
  Adam (per-row elementwise, visible rows have real gradients), not a tuning
  artifact.
what_must_not_be_revived: |
  Any "topology-aware scheduler adds optimizer skipping headroom" claim. The
  visibility skip is the only exact skip signal; topology is a forced-cost
  channel. Reviving requires a NEW independent mechanism that makes a visible
  row's optimizer step provably redundant (none known; would break exactness).
artifacts:
  - vts0/results/room_vts0_result.json
  - vts0/results/bicycle_vts0_result.json
  - vts0/results/room_topology_events.json
  - vts0/results/bicycle_topology_events.json
  - vts0/vts0_trace_driver.py
  - vts0/vts0_analyze.py
prior_art:
  - VOS_DEPENDENCY_GRAPH.json (visibility-only consumer scopes, frozen topology 15K-30K)
  - LAVOS0_REPORT.md (lineage-aware VOS into dynamic phase; avoidable_transition ~0.836 is VISIBILITY-driven, not topology-driven)
