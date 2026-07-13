"""
Benchmark scripts and utilities for 3D Gaussian Splatting renderer evaluation.

This package contains executable scripts for:
  - Scene generation (generate_scene.py): Synthetic 3DGS scene creation
  - Camera path generation (generate_camera_path.py): Standard trajectory presets
  - Dataset management (download_datasets.py): Real-world scene download
  - Phase 1 benchmarking (benchmark_phase1.py): Cross-renderer comparison
  - Phase 2 benchmarking (benchmark_phase2.py): Optimization ablation study
  - Quality validation (validate_quality.py): PSNR/SSIM/LPIPS verification
  - Report generation (gen_report.py, generate_report.py): HTML/Plotly reports

References:
    Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023).
    3D Gaussian Splatting for Real-Time Radiance Field Rendering.
    ACM Transactions on Graphics, 42(4).
"""