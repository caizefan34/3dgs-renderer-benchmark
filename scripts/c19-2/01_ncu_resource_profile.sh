#!/bin/bash
# C19-2 Goal 1: Obtain compiler resource data via ncu profiling.
# Runs ncu on the rasterize kernel for tile_size=16 and tile_size=20.
# Must be run on the A100 machine (mx).
#
# The ncu installation on mx has a section path bug. We work around it with:
#   --section-folder /usr/lib/nsight-compute/sections

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

NCU="ncu"
SECTION_FOLDER="/usr/lib/nsight-compute/sections"
RULE_FOLDER="/usr/lib/nsight-compute/extras/RuleTemplates"

# Target metrics for register/spill analysis
METRICS=(
    "register_count"
    "launch__registers_per_thread"
    "launch__shared_mem_per_block_dynamic"
    "launch__shared_mem_per_block_static"
    "l1tex__local_load_hit_rate"
    "l1tex__local_store_hit_rate"
    "l1tex__local_load_transactions"
    "l1tex__local_store_transactions"
    "l1tex__local_load_sectors"
    "l1tex__local_store_sectors"
    "dram__local_load_transactions"
    "dram__local_store_transactions"
    "sm__warps_active"
    "sm__warps_occupancy"
    "sm__maximum_warps_per_active_cycle"
    "sm__warps_issue_stalled_barrier_per_issue_active"
    "sm__warps_issue_stalled_long_scoreboard_per_issue_active"
    "sm__warps_issue_stalled_short_scoreboard_per_issue_active"
    "sm__warps_issue_stalled_math_pipe_throttle_per_issue_active"
    "sm__warps_issue_stalled_membar_per_issue_active"
    "sm__warps_issue_stalled_memory_dependency_per_issue_active"
    "sm__warps_issue_stalled_not_selected_per_issue_active"
    "sm__warps_issue_stalled_sleeping_per_issue_active"
    "sm__warps_issue_stalled_sync_intrinsic_per_issue_active"
    "sm__warps_issue_stalled_tex_throttle_per_issue_active"
    "sm__warps_issue_stalled_wait_per_issue_active"
    "sm__inst_executed"
)

METRICS_CSV=$(IFS=,; echo "${METRICS[*]}")

for TS in 16 20; do
    echo "=== Profiling tile_size=$TS ==="
    echo "Output: results/phase-c19/ncu_profile_tile${TS}"

    CUDA_VISIBLE_DEVICES=0 \
    HOME=/tmp \
    $NCU --section-folder "$SECTION_FOLDER" \
         --section-folder-recursive "$RULE_FOLDER" \
         --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
         --launch-skip 3 \
         --launch-count 1 \
         --metrics "$METRICS_CSV" \
         --csv \
         -o "results/phase-c19/ncu_profile_tile${TS}" \
         python3 scripts/c19-2/00_reproducibility_baseline.py --tile-size "$TS"

    echo "Done tile_size=$TS"
done

echo "=== NCU profiling complete ==="
