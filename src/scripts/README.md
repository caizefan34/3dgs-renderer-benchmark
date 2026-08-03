# Python Workers (`src/scripts/`)

One-off and pipeline Python scripts invoked directly (often by CI or by the
`benchmark` CLI). Common groups:

- **Dataset / asset prep**: `download_datasets.py`, `generate_scene.py`, `gen_cameras.py`, `generate_camera_path.py`, `prepare_*`
- **Benchmark execution**: `benchmark_phase1.py`, `benchmark_phase2.py`, `run_local_renderer_suite.py`, `run_epic05_*`
- **Collection / validation**: `collect_matrix_result.py`, `collect_compression_result.py`, `validate_artifacts.py`, `validate_benchmark_suite.py`, `validate_official_training.py`, `validate_quality.py`
- **Analysis / reporting**: `analyze_results.py`, `analyze_temporal_sequence.py`, `generate_leaderboard.py`, `generate_matrix_rankings.py`, `generate_plots.py`, `generate_report.py`, `generate_resolution_scaling.py`, `generate_training_report.py`, `check_regressions.py`
- **Compression**: `compress_ply.py`, `collect_compression_result.py`, `audit_compression_visuals.py`

These are Python utilities; Linux shell/environment scripts live in the root
`scripts/` directory instead.
