"""
Reference V1 configuration — canonical Room 30K training parameters.

Semantic label: REFERENCE_V1_ABSGRAD
  Densification uses absgrad=True with grow_grad2d=0.0008 (gsplat AbsGS adaptation).
  This is NOT literal original-3DGS signed-gradient densification (which uses 0.0002).

All other values match official Graphdeco OptimizationParams (arguments/__init__.py L74-100)
at commit 54c035f, with explicit documentation of any gsplat adaptations.
"""

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class ReferenceV1Config:
    """Canonical reference training configuration.

    Every field is explicitly defined. No hidden thresholds in source code.
    """

    # === Scene ===
    scene: str = "room"
    resolution: str = "1080p"  # 1920×1080
    repo_root: str = "/home/liaoyuanjun/3dgs-renderer-benchmark"

    # === Training ===
    iterations: int = 30_000
    seed: int = 42

    # === Learning rates (official OptimizationParams) ===
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.025
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

    # === Densification (official + gsplat adaptation) ===
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densification_interval: int = 100
    # gsplat adaptation: absgrad requires 4x higher threshold.
    # Official: 0.0002 (signed gradient in pixel space)
    # gsplat absgrad: 0.0008 (absolute gradient in pixel space)
    # See: gsplat/strategy/default.py docstring
    densify_grad_threshold: float = 0.0008
    percent_dense: float = 0.01

    # === Pruning (official) ===
    min_opacity: float = 0.005
    max_screen_size: int = 20  # official: size_threshold = 20 if iter > opacity_reset_interval

    # === Opacity reset (official) ===
    opacity_reset_interval: int = 3000

    # === SH progression (official) ===
    sh_degree: int = 3
    sh_progress_interval: int = 1000  # oneupSHdegree every 1000 iters

    # === Loss (official) ===
    lambda_dssim: float = 0.2  # loss = (1-λ)L1 + λ(1-SSIM)

    # === Renderer (gsplat adaptation) ===
    renderer: str = "gsplat"
    gsplat_version: str = "1.5.3"
    tile_size: int = 16
    packed: bool = False
    semantic_label: str = "REFERENCE_V1_ABSGRAD"

    # === Evaluation checkpoints ===
    eval_iterations: tuple = (500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000)

    # === C49/C50/C53 instrumentation checkpoints ===
    instrument_iterations: tuple = (500, 1000, 2000, 5000, 10000, 15000, 20000, 25000)

    # === Model checkpoints (for continuation experiments) ===
    checkpoint_iterations: tuple = (2000, 5000, 10000, 14000, 15000, 30000)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["eval_iterations"] = list(self.eval_iterations)
        d["instrument_iterations"] = list(self.instrument_iterations)
        d["checkpoint_iterations"] = list(self.checkpoint_iterations)
        return d
