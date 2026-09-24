#!/bin/bash
# Create a tarball of all files needed for the audit package from the remote
set -e
cd ~/3dgs-renderer-benchmark

STAGE=/tmp/audit_stage
rm -rf $STAGE
mkdir -p $STAGE

# 1. R0.1 / R0.2 source
mkdir -p $STAGE/baseline/r0.1 $STAGE/baseline/r0.2
cp baseline/r0.1/continuation_runner.py $STAGE/baseline/r0.1/
cp baseline/r0.1/r0.1_analysis.py $STAGE/baseline/r0.1/
cp baseline/r0.2/continuation_runner.py $STAGE/baseline/r0.2/
cp baseline/r0.2/r0.2_analysis.py $STAGE/baseline/r0.2/

# 2. R0.1 results
mkdir -p $STAGE/results/reference_v1/r0.1
for f in c49_gdens_concentration.json c49_gopt_concentration.json c50_true_lag1_gopt.json c53_true_lag1_workload.json final_go_no_go.json gopt_gdens_correlation.json; do
    cp results/reference_v1/r0.1/$f $STAGE/results/reference_v1/r0.1/ 2>/dev/null || echo "MISSING: r0.1/$f"
done

# 3. R0.2 results
mkdir -p $STAGE/results/reference_v1/r0.2
for f in topk_metrics_corrected.json predictive_gradient_mass_coverage.json historical_mask_source_audit.json gdens_gopt_per_gaussian.json backward_signal_availability.json reference_backward_profile.json final_decision.json backward_profile.json; do
    cp results/reference_v1/r0.2/$f $STAGE/results/reference_v1/r0.2/ 2>/dev/null || echo "MISSING: r0.2/$f"
done

# 4. R0.3 results
mkdir -p $STAGE/results/reference_v1/r0.3
for f in visible_conditional_correlations.json visible_gradient_mass_coverage.json visibility_confound_decomposition.json work_gradient_pareto.json signal_availability_cost.json masked_backward_scaling.json final_decision.json; do
    cp results/reference_v1/r0.3/$f $STAGE/results/reference_v1/r0.3/ 2>/dev/null || echo "MISSING: r0.3/$f"
done

# 5. room_30k summary
mkdir -p $STAGE/results/reference_v1/room_30k
for f in provenance.json config.json training_metrics.json timing.json topology_events.json lineage_summary.json c49_gradient_concentration.json c50_temporal_predictability.json c53_workload_statistics.json; do
    cp results/reference_v1/room_30k/$f $STAGE/results/reference_v1/room_30k/ 2>/dev/null || echo "MISSING: room_30k/$f"
done

# 6. Configs
mkdir -p $STAGE/configs/reference_v1
cp configs/reference_v1/room_30k.yaml $STAGE/configs/reference_v1/ 2>/dev/null || echo "MISSING: configs/reference_v1/room_30k.yaml"

# 7. Tests
mkdir -p $STAGE/tests/reference_v1
cp tests/reference_v1/test_reference_v1.py $STAGE/tests/reference_v1/ 2>/dev/null || echo "MISSING: tests/reference_v1/test_reference_v1.py"

# 8. C51 scripts
mkdir -p $STAGE/scripts/phase-c51-stage4a $STAGE/scripts/phase-c51-stage4b $STAGE/scripts/phase-c51 $STAGE/scripts/phase-c51r
cp scripts/phase-c51-stage4a/patch_cuda.py $STAGE/scripts/phase-c51-stage4a/ 2>/dev/null || echo "MISSING: patch_cuda.py"
cp scripts/phase-c51-stage4b/canonical_training.py $STAGE/scripts/phase-c51-stage4b/ 2>/dev/null || echo "MISSING: canonical_training.py"
cp scripts/phase-c51-stage4b/measure_recall.py $STAGE/scripts/phase-c51-stage4b/ 2>/dev/null || echo "MISSING: measure_recall.py"
cp scripts/phase-c51/simulated_sparse_backward.py $STAGE/scripts/phase-c51/ 2>/dev/null || echo "MISSING: simulated_sparse_backward.py"
cp scripts/phase-c51r/run_experiment.py $STAGE/scripts/phase-c51r/ 2>/dev/null || echo "MISSING: phase-c51r/run_experiment.py"
cp scripts/phase-c51-stage4a/patch_bwd_kernel_only.py $STAGE/scripts/phase-c51-stage4a/ 2>/dev/null || echo "MISSING: patch_bwd_kernel_only.py"

# 9. gsplat source snapshot - relevant files only
GSPLAT_SRC=/tmp/gsplat_baseline/gsplat-1.5.3/gsplat
mkdir -p $STAGE/external_source_snapshot/gsplat_c51/cuda/csrc
mkdir -p $STAGE/external_source_snapshot/gsplat_c51/cuda/include

# Python files
cp $GSPLAT_SRC/rendering.py $STAGE/external_source_snapshot/gsplat_c51/
cp $GSPLAT_SRC/__init__.py $STAGE/external_source_snapshot/gsplat_c51/
cp $GSPLAT_SRC/version.py $STAGE/external_source_snapshot/gsplat_c51/
cp $GSPLAT_SRC/_helper.py $STAGE/external_source_snapshot/gsplat_c11/ 2>/dev/null || true
cp $GSPLAT_SRC/_helper.py $STAGE/external_source_snapshot/gsplat_c51/ 2>/dev/null || true

# cuda Python wrappers
cp $GSPLAT_SRC/cuda/_wrapper.py $STAGE/external_source_snapshot/gsplat_c51/cuda/
cp $GSPLAT_SRC/cuda/_wrapper.py.orig_stage4a $STAGE/external_source_snapshot/gsplat_c51/cuda/ 2>/dev/null || true
cp $GSPLAT_SRC/cuda/_backend.py $STAGE/external_source_snapshot/gsplat_c51/cuda/
cp $GSPLAT_SRC/cuda/_torch_impl.py $STAGE/external_source_snapshot/gsplat_c51/cuda/
cp $GSPLAT_SRC/cuda/ext.cpp $STAGE/external_source_snapshot/gsplat_c51/cuda/
cp $GSPLAT_SRC/cuda/__init__.py $STAGE/external_source_snapshot/gsplat_c51/cuda/

# csrc - only .cu, .cuh, .h, .cpp (no third_party)
for f in $GSPLAT_SRC/cuda/csrc/*.cu $GSPLAT_SRC/cuda/csrc/*.cuh $GSPLAT_SRC/cuda/csrc/*.h $GSPLAT_SRC/cuda/csrc/*.cpp; do
    cp "$f" $STAGE/external_source_snapshot/gsplat_c51/cuda/csrc/ 2>/dev/null || true
done

# include files
for f in $GSPLAT_SRC/cuda/include/*.h $GSPLAT_SRC/cuda/include/*.cuh; do
    cp "$f" $STAGE/external_source_snapshot/gsplat_c51/cuda/include/ 2>/dev/null || true
done
# include orig
cp $GSPLAT_SRC/cuda/include/Ops.h.orig_stage4a $STAGE/external_source_snapshot/gsplat_c51/cuda/include/ 2>/dev/null || true

# 10. Also fetch the r0.3 window-level JSON files (visible_fraction etc)
for w in 2000 5000 10000 14000; do
    mkdir -p $STAGE/results/reference_v1/r0.3/window_$w
    for f in visible_fraction.json visible_conditional_correlations.json visible_gradient_mass_coverage.json all_gaussian_coverage.json work_gradient_pareto.json provenance.json; do
        cp results/reference_v1/r0.3/window_$w/$f $STAGE/results/reference_v1/r0.3/window_$w/ 2>/dev/null || true
    done
done

# Create tarball
cd /tmp
tar czf audit_stage.tar.gz -C $STAGE .
ls -lh /tmp/audit_stage.tar.gz
echo "STAGE_DIR=$STAGE"
