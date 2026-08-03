# Tests

`unittest` suite (no pytest dependency). Run:

```bash
python -m unittest discover -s tests -v
```

CUDA-specific HiGS tests skip cleanly when the CUDA extension is unavailable;
the rest are CPU-safe.

## Inventory by area
- **HiGS trainability**: `test_higs_trainable.py`, `test_higs_frozen.py`, `test_higs_dynamic.py`, `test_higs_native_backward.py`
- **Benchmark CLI / matrix**: `test_benchmark_cli.py`, `test_benchmark_matrix.py`, `test_benchmark_suite.py`, `test_training_matrix.py`, `test_temporal_matrix.py`, `test_linux_tier_a_matrix.py`, `test_prepare_suite_case.py`, `test_renderer_research_experiments.py`
- **Renderers / adapters**: `test_renderers.py`, `test_candidate_renderers.py`, `test_candidate_smoke_matrix.py`, `test_flashgs_renderer.py`, `test_adapters.py`
- **Quality / compression**: `test_quality.py`, `test_compression_artifact.py`, `test_compression_candidates.py`, `test_compression_result.py`, `test_compression_visual_audit.py`, `test_merge_compression_sessions.py`
- **Data / pipeline**: `test_dataset_pipeline.py`, `test_scene.py`, `test_official_training_policy.py`, `test_resolution_scaling.py`, `test_temporal_sequence.py`
- **Results / docs / environment**: `test_results.py`, `test_documentation.py`, `test_leaderboard_platform.py`, `test_training_report.py`, `test_goal_results_presentation.py`, `test_nvml.py`, `test_wait_for_json_status.py`, `test_evaluation.py`, `test_local_renderer_suite.py`, `test_collect_matrix_result.py`
