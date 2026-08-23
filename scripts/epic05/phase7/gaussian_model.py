"""
3DGS Gaussian model with full training capability.

Provides:
- SfM initialization from PLY point cloud
- Parameter activations (exp for scales, sigmoid for opacity, normalize for quats)
- Densification (clone/split based on positional gradient magnitude)
- Pruning (opacity-based removal with reset)
- SH degree resizing / progressive scheduling
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianModel(nn.Module):
    """Full 3DGS Gaussian model with topological operations.

    Follows the original 3DGS training scheme [Kerbl et al., 2023]:
      - 5 parameter groups: xyz, rotations (quats), scales (log), opacity (logit), SH
      - Densification via gradient-magnitude threshold (clone + split)
      - Pruning via opacity threshold
      - Progressive SH degree increase
    """

    def __init__(
        self,
        num_points: int,
        sh_degree: int = 3,
        max_sh_degree: int = 3,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_points = num_points
        self.sh_degree = sh_degree  # current degree (may increase during training)
        self.max_sh_degree = max_sh_degree
        self.device = device
        self.num_sh_coeffs = (sh_degree + 1) ** 2

        # Activation cache (set during forward)
        self._activated = {}
        self._xyz_grad_accum = None  # positional gradient accumulation for densification
        self._denf_steps = 0

    def init_from_sfm(
        self,
        xyz: torch.Tensor,
        opacity_logit: torch.Tensor | None = None,
        scales_log: torch.Tensor | None = None,
        rotations_raw: torch.Tensor | None = None,
        shs: torch.Tensor | None = None,
    ):
        """Initialize from an SfM / PLY point cloud.

        If NULL values are provided, sensible defaults are used based on
        the original 3DGS initialization scheme.
        """
        N = xyz.shape[0]
        self.num_points = N

        self.xyz = nn.Parameter(xyz.to(self.device).contiguous())

        # Opacity: logit space (mean ~0.1 after sigmoid)
        if opacity_logit is not None:
            self.opacity = nn.Parameter(opacity_logit.to(self.device).contiguous())
        else:
            self.opacity = nn.Parameter(
                torch.logit(torch.full((N, 1), 0.1, device=self.device))
            )

        # Scales: log-space
        if scales_log is not None:
            self.scales = nn.Parameter(scales_log.to(self.device).contiguous())
        else:
            # mean scale ~ 0.01 (world-units)
            self.scales = nn.Parameter(
                torch.log(torch.full((N, 3), 0.01, device=self.device))
            )

        # Rotations: 4D raw (will be normalized in forward)
        if rotations_raw is not None:
            self.rotations = nn.Parameter(rotations_raw.to(self.device).contiguous())
        else:
            self.rotations = nn.Parameter(
                torch.zeros(N, 4, device=self.device)
            )
            self.rotations.data[:, 0] = 1.0  # identity quaternion

        # SH coefficients
        if shs is not None:
            self.shs = nn.Parameter(shs.to(self.device).contiguous())
            sh_degree = round(math.sqrt(shs.shape[1]) - 1)
            self.sh_degree = int(sh_degree)
        else:
            self.shs = nn.Parameter(
                torch.zeros(N, self.num_sh_coeffs, 3, device=self.device)
            )

        self.num_points = N

    def forward(self) -> Dict[str, torch.Tensor]:
        """Return activated parameters for the renderer."""
        self._activated = {
            "xyz": self.xyz,
            "rotations": F.normalize(self.rotations, dim=-1).contiguous(),
            "scales": torch.exp(self.scales).contiguous(),
            "opacity": torch.sigmoid(self.opacity).squeeze(-1).contiguous(),
            "shs": self.shs.contiguous(),
            "num_points": self.xyz.shape[0],
            "sh_degree": self.sh_degree,
        }
        return self._activated

    def set_sh_degree(self, degree: int):
        """Resize SH coefficients for progressive degree increase.

        When degree increases, new higher-order coefficients are zero-initialized.
        When degree decreases (shouldn't in normal training), coefficients are truncated.
        """
        if degree == self.sh_degree:
            return
        new_coeffs = (degree + 1) ** 2
        old_coeffs = self.shs.shape[1]

        if new_coeffs > old_coeffs:
            # Pad with zeros
            pad = torch.zeros(
                self.shs.shape[0], new_coeffs - old_coeffs, 3,
                dtype=self.shs.dtype, device=self.shs.device
            )
            new_shs = torch.cat([self.shs.detach(), pad], dim=1)
            self.shs = nn.Parameter(new_shs)
        else:
            self.shs = nn.Parameter(self.shs[:, :new_coeffs].clone())

        self.sh_degree = degree
        self.num_sh_coeffs = new_coeffs
        # Reset accumulation after SH resize
        self._xyz_grad_accum = None
        self._denf_steps = 0

    # ---- Densification -------------------------------------------------------

    @torch.no_grad()
    def accumulate_positional_gradient(self):
        """Accumulate the gradient of xyz for densification thresholding."""
        if self.xyz.grad is None:
            return
        grad = self.xyz.grad.detach()
        grad_norm = grad.norm(dim=-1)
        if self._xyz_grad_accum is None or self._xyz_grad_accum.shape != grad_norm.shape:
            self._xyz_grad_accum = grad_norm
            self._denf_steps = 1
        else:
            self._xyz_grad_accum = self._xyz_grad_accum + grad_norm
            self._denf_steps += 1

    @torch.no_grad()
    def densification(
        self,
        grad_threshold: float = 2e-4,
        clone_max_screen_size: float = 100.0,
        split_max_screen_size: float = 100.0,
    ) -> Dict[str, int]:
        """Clone Gaussians with high gradient AND small scale; split those with high gradient AND large scale.

        Returns counts of clones and splits performed.
        """
        if self._xyz_grad_accum is None or self._denf_steps == 0:
            return {"cloned": 0, "split": 0, "removed": 0}

        avg_grad = self._xyz_grad_accum / self._denf_steps
        high_grad_mask = avg_grad >= grad_threshold
        self._xyz_grad_accum = None
        self._denf_steps = 0

        scales_activated = torch.exp(self.scales).detach()
        median_scale = scales_activated.median(dim=0).values
        is_small = (scales_activated <= median_scale).all(dim=-1)
        clone_mask = high_grad_mask & is_small
        split_mask = high_grad_mask & (~is_small)

        clone_count, split_count = clone_mask.sum().item(), split_mask.sum().item()

        if clone_count == 0 and split_count == 0:
            return {"cloned": 0, "split": 0, "removed": 0}

        # CRITICAL: Extract all data BEFORE any parameter modifications
        original_xyz = self.xyz.detach()
        original_rotations = self.rotations.detach()
        original_scales = self.scales.detach()
        original_opacity = self.opacity.detach()
        original_shs = self.shs.detach()

        new_xyz_parts = []
        new_rotations_parts = []
        new_scales_parts = []
        new_opacity_parts = []
        new_shs_parts = []

        # Clone
        if clone_count > 0:
            cloned_xyz = original_xyz[clone_mask]
            noise = torch.randn_like(cloned_xyz) * 0.01 * torch.exp(original_scales[clone_mask])
            new_xyz_parts.append(cloned_xyz + noise)
            new_rotations_parts.append(original_rotations[clone_mask])
            new_scales_parts.append(original_scales[clone_mask])
            new_opacity_parts.append(original_opacity[clone_mask])
            new_shs_parts.append(original_shs[clone_mask])

        # Split
        if split_count > 0:
            split_xyz = original_xyz[split_mask]
            split_rotations = original_rotations[split_mask]
            split_scales = original_scales[split_mask] - math.log(2.0)  # halve scale
            split_opacity = original_opacity[split_mask]
            split_shs = original_shs[split_mask]

            # Duplicate
            new_xyz_parts.append(torch.cat([split_xyz, split_xyz], dim=0))
            new_rotations_parts.append(torch.cat([split_rotations, split_rotations], dim=0))
            new_scales_parts.append(torch.cat([split_scales, split_scales], dim=0))
            new_opacity_parts.append(torch.cat([split_opacity, split_opacity], dim=0))
            new_shs_parts.append(torch.cat([split_shs, split_shs], dim=0))

        # Concatenate all new Gaussians
        new_xyz = torch.cat(new_xyz_parts, dim=0)
        new_rotations = torch.cat(new_rotations_parts, dim=0)
        new_scales = torch.cat(new_scales_parts, dim=0)
        new_opacity = torch.cat(new_opacity_parts, dim=0)
        new_shs = torch.cat(new_shs_parts, dim=0)

        # Add noise for split positions
        if split_count > 0:
            split_noise = torch.randn_like(new_xyz[-split_count * 2:]) * 0.0025
            new_xyz[-split_count * 2:] = new_xyz[-split_count * 2:] + split_noise

        # Append all at once
        self.xyz = nn.Parameter(
            torch.cat([self.xyz, new_xyz], dim=0).contiguous()
        )
        self.rotations = nn.Parameter(
            torch.cat([self.rotations, new_rotations], dim=0).contiguous()
        )
        self.scales = nn.Parameter(
            torch.cat([self.scales, new_scales], dim=0).contiguous()
        )
        self.opacity = nn.Parameter(
            torch.cat([self.opacity, new_opacity], dim=0).contiguous()
        )
        self.shs = nn.Parameter(
            torch.cat([self.shs, new_shs], dim=0).contiguous()
        )
        self.num_points = self.xyz.shape[0]
        # Reset accumulation
        self._xyz_grad_accum = None
        self._denf_steps = 0

        return {"cloned": clone_count, "split": split_count, "removed": 0}

    @torch.no_grad()
    def prune(self, opacity_threshold: float = 0.005) -> int:
        """Remove Gaussians with opacity below threshold.

        Also resets opacity for surviving Gaussians near the threshold
        to keep some alive that might otherwise be pruned.
        """
        opacities = torch.sigmoid(self.opacity).detach().squeeze(-1)
        prune_mask = opacities < opacity_threshold
        removed = prune_mask.sum().item()

        if removed == 0:
            return 0

        keep_mask = ~prune_mask
        self.xyz = nn.Parameter(self.xyz[keep_mask].contiguous())
        self.rotations = nn.Parameter(self.rotations[keep_mask].contiguous())
        self.scales = nn.Parameter(self.scales[keep_mask].contiguous())
        self.opacity = nn.Parameter(self.opacity[keep_mask].contiguous())
        self.shs = nn.Parameter(self.shs[keep_mask].contiguous())
        self.num_points = self.xyz.shape[0]

        # Reset accumulation after topology change
        self._xyz_grad_accum = None
        self._denf_steps = 0

        return removed

    @torch.no_grad()
    def prune_and_reset(
        self,
        opacity_threshold: float = 0.005,
        reset_interval: int = 3000,
        current_step: int = 0,
    ) -> int:
        """Prune low-opacity Gaussians and optionally reset opacity of some survivors."""
        removed = self.prune(opacity_threshold)

        # Reset opacity for Gaussians near the threshold
        if current_step > 0 and current_step % reset_interval == 0:
            opacities = torch.sigmoid(self.opacity).detach().squeeze(-1)
            near_threshold = (opacities < opacity_threshold * 10) & (opacities >= opacity_threshold)
            reset_count = near_threshold.sum().item()
            if reset_count > 0:
                reset_val = torch.logit(torch.full((reset_count, 1), opacity_threshold * 2, device=self.device))
                self.opacity.data[near_threshold] = reset_val

        return removed

    # ---- Internal helpers ----------------------------------------------------

    def _clone_params(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "rotations": self.rotations[mask].detach().contiguous(),
            "scales": self.scales[mask].detach().contiguous(),
            "opacity": self.opacity[mask].detach().contiguous(),
            "shs": self.shs[mask].detach().contiguous(),
        }

    def _append_gaussians(self, new_xyz: torch.Tensor, features: Dict[str, torch.Tensor]):
        self.xyz = nn.Parameter(
            torch.cat([self.xyz, new_xyz], dim=0).contiguous()
        )
        for key, tensor in features.items():
            param = getattr(self, key)
            setattr(self, key, nn.Parameter(
                torch.cat([param, tensor], dim=0).contiguous()
            ))
        self.num_points = self.xyz.shape[0]
        # Reset accumulation after densification
        self._xyz_grad_accum = None
        self._denf_steps = 0

    @torch.no_grad()
    def get_optimizer_param_groups(self, config: Dict[str, float]) -> list:
        """Return parameter groups for Adam optimizer with per-group LR.

        Config keys (same as original 3DGS):
            lr_xyz, lr_rotation, lr_scaling, lr_opacity, lr_sh
        """
        return [
            {"params": [self.xyz], "lr": config.get("lr_xyz", 1.6e-4)},
            {"params": [self.rotations], "lr": config.get("lr_rotation", 1e-3)},
            {"params": [self.scales], "lr": config.get("lr_scaling", 5e-3)},
            {"params": [self.opacity], "lr": config.get("lr_opacity", 5e-2)},
            {"params": [self.shs], "lr": config.get("lr_sh", 2.5e-3)},
        ]

    @torch.no_grad()
    def get_checkpoint_state(self) -> Dict:
        return {
            "xyz": self.xyz.detach().cpu(),
            "rotations": self.rotations.detach().cpu(),
            "scales": self.scales.detach().cpu(),
            "opacity": self.opacity.detach().cpu(),
            "shs": self.shs.detach().cpu(),
            "sh_degree": self.sh_degree,
            "num_points": self.num_points,
        }

    @classmethod
    def from_checkpoint_state(cls, state: Dict, device: str = "cuda") -> "GaussianModel":
        model = cls(num_points=state["num_points"], sh_degree=state["sh_degree"], device=device)
        model.xyz = nn.Parameter(state["xyz"].to(device))
        model.rotations = nn.Parameter(state["rotations"].to(device))
        model.scales = nn.Parameter(state["scales"].to(device))
        model.opacity = nn.Parameter(state["opacity"].to(device))
        model.shs = nn.Parameter(state["shs"].to(device))
        return model
