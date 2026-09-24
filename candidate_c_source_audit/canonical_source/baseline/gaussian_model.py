"""
Reference V1 GaussianModel — Faithful Graphdeco 3DGS training semantics.

Pinned official source: graphdeco-inria/gaussian-splatting @ 54c035f
Adapted for gsplat 1.5.3 rendering backend (not diff_gaussian_rasterization).

Key semantic choices (each documented with official source reference):
  - Split: parent IS removed after adding N=2 children (official: densify_and_split L409-433)
  - Clone: exact parameter copy, NO noise (official: densify_and_clone L435-450)
  - Split scale: s_child = s_parent / (0.8 * N) in activated space (official L423)
  - Split position: rotation @ normal(0, scale) + parent_xyz (official L418-422)
  - Gradient: view-space mean2D gradient (official: add_densification_stats L471-473)
  - Selection: percent_dense * scene_extent (official L416,439)
  - Optimizer: ONE persistent Adam, state migrated across topology changes
    (official: cat_tensors_to_optimizer L366-386, _prune_optimizer L331-347)
  - SH: fixed max tensor, only active_sh_degree increments (official: oneupSHdegree L145-147)
  - Prune: opacity + screen-size + world-size (official: densify_and_prune L460-464)
  - Opacity reset: global min(opacity, 0.01) (official: reset_opacity L258-261)

gsplat adaptation:
  - SH stored as single [N, K, 3] tensor (gsplat) instead of separate dc/rest
  - means2d from gsplat meta dict with retain_grad() for view-space gradient
  - radii from gsplat meta for visibility filter
  - No exposure/exposure_optimizer (not part of core 3DGS semantics)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianModel:
    """Reference 3DGS model with official Graphdeco training semantics.

    All topology operations preserve optimizer state via cat_tensors_to_optimizer
    and _prune_optimizer. A single Adam optimizer is created in training_setup
    and persists throughout training.
    """

    def __init__(self, max_sh_degree: int = 3):
        self.active_sh_degree = 0
        self.max_sh_degree = max_sh_degree
        self.num_sh_coeffs = (max_sh_degree + 1) ** 2

        # Parameters (nn.Parameter, set in create_from_pcd / load_ply)
        self._xyz = torch.empty(0)
        self._shs = torch.empty(0)        # [N, K, 3] — fixed max tensor
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # Densification stats
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.tmp_radii = None

        # Explicit Gaussian identity tracking (R0.2 Part H)
        # Does not affect rendering; purely metadata for lineage tracking.
        self._gaussian_ids = torch.empty(0, dtype=torch.long, device="cuda")
        self._next_gaussian_id = 0

        # Config (set in training_setup)
        self.optimizer = None
        self.percent_dense = 0.01
        self.spatial_lr_scale = 1.0
        self.xyz_scheduler_args = None

    # ---- Activations (match official) ---------------------------------------

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return F.normalize(self._rotation, dim=-1)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity).squeeze(-1)  # 1D for gsplat

    @property
    def get_features(self):
        """Return SH coefficients up to active_sh_degree."""
        n_active = (self.active_sh_degree + 1) ** 2
        return self._shs[:, :n_active, :].contiguous()

    @property
    def get_sh_degree(self):
        return self.active_sh_degree

    # ---- SH progression (official: oneupSHdegree L145-147) ------------------

    def oneupSHdegree(self):
        """Increment active SH degree. Does NOT create new Parameter."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ---- Initialization (official: create_from_pcd L149-176) ----------------

    def create_from_pcd(self, pcd_data: dict, spatial_lr_scale: float):
        """Initialize from SfM point cloud data.

        Args:
            pcd_data: dict with keys 'xyz' [N,3], 'shs' [N,K,3] or None,
                      'opacity' [N,1] or None, 'scales' [N,3] or None,
                      'rotations' [N,4] or None
            spatial_lr_scale: scene extent for LR scheduling
        """
        self.spatial_lr_scale = spatial_lr_scale
        N = pcd_data["xyz"].shape[0]

        fused_point_cloud = pcd_data["xyz"].float().cuda()
        opacities = pcd_data.get("opacity")
        if opacities is not None:
            opacities = opacities.float().cuda().squeeze(-1)  # 1D for gsplat
        else:
            # Official: inverse_sigmoid(0.1)
            opacities = torch.logit(torch.full((N,), 0.1, device="cuda"))  # 1D

        scales = pcd_data.get("scales")
        if scales is not None:
            scales = scales.float().cuda()
        else:
            # Official: log(sqrt(dist2)) repeated 3x
            # Use KNN distance as default scale
            dist2 = self._compute_knn_dist(fused_point_cloud)
            scales = torch.log(torch.sqrt(dist2)).unsqueeze(-1).repeat(1, 3)

        rotations = pcd_data.get("rotations")
        if rotations is not None:
            rotations = rotations.float().cuda()
        else:
            rotations = torch.zeros(N, 4, device="cuda")
            rotations[:, 0] = 1.0  # identity quaternion

        shs = pcd_data.get("shs")
        if shs is not None:
            shs = shs.float().cuda()
            # Ensure shape is [N, K, 3] with K = max_sh_coeffs
            if shs.shape[1] < self.num_sh_coeffs:
                pad = torch.zeros(N, self.num_sh_coeffs - shs.shape[1], 3,
                                  device="cuda")
                shs = torch.cat([shs, pad], dim=1)
            elif shs.shape[1] > self.num_sh_coeffs:
                shs = shs[:, :self.num_sh_coeffs, :]
        else:
            shs = torch.zeros(N, self.num_sh_coeffs, 3, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._shs = nn.Parameter(shs.contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(rotations.contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.contiguous().requires_grad_(True))

        self.max_radii2D = torch.zeros((N,), device="cuda")
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.denom = torch.zeros((N, 1), device="cuda")

        # Initialize explicit Gaussian IDs (R0.2 Part H)
        self._gaussian_ids = torch.arange(N, dtype=torch.long, device="cuda")
        self._next_gaussian_id = N

        print(f"Number of points at initialisation : {N}")

    @staticmethod
    def _compute_knn_dist(points: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Compute KNN distance for default scale initialization (memory-efficient).

        Uses chunked computation to avoid OOM on large point clouds.
        Official uses simple_knn._C.distCUDA2; we use a chunked fallback.
        """
        N = points.shape[0]
        if N < 4:
            return torch.full((N,), 0.01, device=points.device)
        device = points.device
        chunk_size = 1024
        dist2 = torch.zeros(N, device=device)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            # [chunk, N] distances
            chunk_dists = torch.cdist(points[start:end], points)
            # Exclude self (set diagonal-ish to inf)
            for i in range(start, end):
                chunk_dists[i - start, i] = float('inf')
            # k-th nearest neighbor (k=3, so index 2 since 0 is excluded self)
            knn = chunk_dists.topk(k, dim=1, largest=False)
            dist2[start:end] = knn.values[:, -1].clamp_min(1e-7)
        return dist2

    # ---- Training setup (official: training_setup L178-206) -----------------

    def training_setup(self, config: dict):
        """Create ONE persistent optimizer with per-group LR.

        Config keys (matching official OptimizationParams):
            position_lr_init, position_lr_final, position_lr_delay_mult,
            position_lr_max_steps, feature_lr, opacity_lr, scaling_lr,
            rotation_lr, percent_dense
        """
        self.percent_dense = config["percent_dense"]
        N = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.denom = torch.zeros((N, 1), device="cuda")

        lr_init = config["position_lr_init"] * self.spatial_lr_scale
        lr_final = config["position_lr_final"] * self.spatial_lr_scale

        param_groups = [
            {"params": [self._xyz], "lr": lr_init, "name": "xyz"},
            {"params": [self._shs], "lr": config["feature_lr"], "name": "shs"},
            {"params": [self._opacity], "lr": config["opacity_lr"], "name": "opacity"},
            {"params": [self._scaling], "lr": config["scaling_lr"], "name": "scaling"},
            {"params": [self._rotation], "lr": config["rotation_lr"], "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        # Exponential LR schedule for xyz
        self.xyz_scheduler_args = self._make_expon_lr_func(
            lr_init=lr_init,
            lr_final=lr_final,
            lr_delay_mult=config.get("position_lr_delay_mult", 0.01),
            max_steps=config.get("position_lr_max_steps", 30000),
        )

    @staticmethod
    def _make_expon_lr_func(lr_init, lr_final, lr_delay_mult=0.01, max_steps=30000):
        """Exponential LR schedule matching official get_expon_lr_func."""
        def helper(step):
            if lr_delay_mult > 0:
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * math.sin(
                    0.5 * math.pi * min(step / max_steps, 1.0)
                )
            else:
                delay_rate = 1.0
            t = min(step / max_steps, 1.0)
            log_lerp = math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
            return delay_rate * log_lerp
        return helper

    def update_learning_rate(self, iteration: int):
        """Update xyz LR per exponential schedule (official: update_learning_rate L213-223)."""
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = self.xyz_scheduler_args(iteration)
                return

    # ---- Densification stats (official: add_densification_stats L471-473) ----

    def add_densification_stats(self, means2d: torch.Tensor, visibility_filter: torch.Tensor,
                                 width: int = None, height: int = None):
        """Accumulate view-space mean2D gradient for densification.

        Official:
            xyz_gradient_accum[update_filter] += norm(viewspace_point_tensor.grad[update_filter,:2])
            denom[update_filter] += 1

        gsplat adaptation: 
        1. Use absgrad (absolute gradient) instead of .grad (signed). gsplat's .grad 
           produces tiny values due to per-pixel gradient cancellation.
        2. Scale from normalized [-1,1] screen space to pixel space, matching 
           diff_gaussian_rasterization's pixel-space gradient.
           gsplat DefaultStrategy does: grads[...,0] *= width/2, grads[...,1] *= height/2
        """
        # Prefer absgrad (absolute gradient, populated with absgrad=True)
        grad = getattr(means2d, "absgrad", None)
        if grad is None:
            grad = means2d.grad
        if grad is None:
            return
        # grad is [1, N, 2] — slice to [N, 2]
        if grad.dim() == 3:
            grad = grad[0]  # [N, 2]
        # Scale from normalized [-1,1] to pixel space (gsplat adaptation)
        if width is not None and height is not None:
            grad = grad.clone()  # avoid in-place on grad tensor
            grad[:, 0] *= width / 2.0
            grad[:, 1] *= height / 2.0
        grad_2d = grad[visibility_filter, :2]  # [N_vis, 2]
        grad_norm = torch.norm(grad_2d, dim=-1, keepdim=True)  # [N_vis, 1]
        self.xyz_gradient_accum[visibility_filter] += grad_norm
        self.denom[visibility_filter] += 1

    # ---- Optimizer state migration (official L331-386) ----------------------

    def _prune_optimizer(self, mask: torch.Tensor) -> Dict[str, nn.Parameter]:
        """Remove pruned rows from optimizer state. Survivors keep exp_avg, exp_avg_sq.

        Official L331-347: stored_state["exp_avg"] = stored_state["exp_avg"][mask]
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict: dict) -> Dict[str, nn.Parameter]:
        """Append new tensors. Survivor state preserved, new child state = zeros.

        Official L366-386:
            exp_avg = cat(exp_avg, zeros_like(extension))
            exp_avg_sq = cat(exp_avg_sq, zeros_like(extension))
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                exp_avg = stored_state["exp_avg"]
                exp_avg_sq = stored_state["exp_avg_sq"]
                # Debug: check dimension match
                if exp_avg.dim() != extension_tensor.dim():
                    print(f"  [WARN cat] {group['name']}: exp_avg {exp_avg.shape} (dim={exp_avg.dim()}) "
                          f"vs ext {extension_tensor.shape} (dim={extension_tensor.dim()})")
                    # Fix: reshape to match
                    if exp_avg.dim() < extension_tensor.dim():
                        exp_avg = exp_avg.unsqueeze(-1)
                    elif exp_avg.dim() > extension_tensor.dim():
                        extension_tensor = extension_tensor.unsqueeze(-1)
                if exp_avg_sq.dim() != extension_tensor.dim():
                    if exp_avg_sq.dim() < extension_tensor.dim():
                        exp_avg_sq = exp_avg_sq.unsqueeze(-1)
                    elif exp_avg_sq.dim() > extension_tensor.dim():
                        extension_tensor = extension_tensor.unsqueeze(-1)

                stored_state["exp_avg"] = torch.cat(
                    (exp_avg, torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (exp_avg_sq, torch.zeros_like(extension_tensor)), dim=0
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str) -> Dict[str, nn.Parameter]:
        """Replace a parameter tensor (for opacity reset). Resets moment rows.

        Official L316-329:
            stored_state["exp_avg"] = zeros_like(tensor)
            stored_state["exp_avg_sq"] = zeros_like(tensor)
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # ---- Densification postfix (official L388-407) --------------------------

    def densification_postfix(self, new_xyz, new_shs, new_opacities,
                               new_scaling, new_rotation, new_tmp_radii=None):
        """Append new Gaussians and reset accumulators.

        Official L388-407.
        """
        d = {
            "xyz": new_xyz,
            "shs": new_shs,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._shs = optimizable_tensors["shs"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.tmp_radii is not None and new_tmp_radii is not None:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0],), device="cuda")

        # Propagate explicit Gaussian IDs (R0.2 Part H)
        # New Gaussians (clone children or split children) get fresh IDs.
        # Survivors retain their IDs (already in self._gaussian_ids).
        n_new = new_xyz.shape[0]
        new_ids = torch.arange(
            self._next_gaussian_id,
            self._next_gaussian_id + n_new,
            dtype=torch.long, device="cuda"
        )
        self._gaussian_ids = torch.cat([self._gaussian_ids, new_ids])
        self._next_gaussian_id += n_new

    # ---- Prune (official L349-364) ------------------------------------------

    def prune_points(self, mask: torch.Tensor):
        """Remove Gaussians where mask=True. Preserve survivor optimizer state.

        Official L349-364.
        """
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._shs = optimizable_tensors["shs"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

        # Filter explicit Gaussian IDs (R0.2 Part H)
        # Pruned Gaussians die; their IDs are removed.
        self._gaussian_ids = self._gaussian_ids[valid_points_mask]

    # ---- Clone (official L435-450) ------------------------------------------

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone high-gradient, small-scale Gaussians.

        Official L435-450: exact parameter copy, NO noise. Parent remains.
        ΔN = +N_clone
        """
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_shs = self._shs[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]  # 1D
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask] if self.tmp_radii is not None else None

        n_cloned = selected_pts_mask.sum().item()
        self.densification_postfix(new_xyz, new_shs, new_opacities,
                                    new_scaling, new_rotation, new_tmp_radii)
        return n_cloned

    # ---- Split (official L409-433) ------------------------------------------

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """Split high-gradient, large-scale Gaussians. Parent IS removed.

        Official L409-433:
          1. Generate N children with sampled positions
          2. Append children via densification_postfix
          3. Remove parent via prune_points

        Scale: s_child = s_parent / (0.8 * N) in activated space
        Position: rotation @ normal(0, scale) + parent_xyz
        ΔN = +N_split (net: +1 per parent for N=2)

        gsplat adaptation: tmp_radii not tracked (use radii from next render)
        """
        n_init_points = self._xyz.shape[0]
        padded_grad = torch.zeros((n_init_points,), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        n_split = selected_pts_mask.sum().item()
        if n_split == 0:
            return 0

        # Sample child positions (official L418-422)
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)  # [N*n_split, 3]
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = self._build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
                  self._xyz[selected_pts_mask].repeat(N, 1)

        # Scale: inverse_activation(scale / (0.8 * N)) = log(scale / 1.6) for N=2
        new_scaling = torch.log(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )

        # Inherit rotation, SH, opacity
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_shs = self._shs[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N)  # 1D
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N) if self.tmp_radii is not None else None

        self.densification_postfix(new_xyz, new_shs, new_opacity,
                                    new_scaling, new_rotation, new_tmp_radii)

        # Remove split parents (official L432-433)
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * n_split, device="cuda", dtype=bool)
        ))
        self.prune_points(prune_filter)

        return n_split

    @staticmethod
    def _build_rotation(quats: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices (official: build_rotation)."""
        q = F.normalize(quats, dim=-1)
        r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.zeros(q.shape[0], 3, 3, device=q.device)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - r * z)
        R[:, 0, 2] = 2 * (x * z + r * y)
        R[:, 1, 0] = 2 * (x * y + r * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - r * x)
        R[:, 2, 0] = 2 * (x * z - r * y)
        R[:, 2, 1] = 2 * (y * z + r * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    # ---- Densify and prune (official L452-469) ------------------------------

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size,
                           radii: torch.Tensor):
        """Execute clone, split, then prune in one transaction.

        Official L452-469.
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        n_cloned = self.densify_and_clone(grads, max_grad, extent)
        n_split = self.densify_and_split(grads, max_grad, extent)
        self.tmp_radii = None  # Reset after densify

        # Prune (official L460-465)
        prune_mask = self.get_opacity < min_opacity  # 1D, no squeeze needed
        pruned_opacity = prune_mask.sum().item()

        pruned_screen = 0
        pruned_world = 0
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            pruned_screen = (big_points_vs & ~prune_mask).sum().item()
            pruned_world = (big_points_ws & ~(prune_mask | big_points_vs)).sum().item()
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )

        total_pruned = prune_mask.sum().item()
        if total_pruned > 0:
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        return {
            "cloned": n_cloned,
            "split": n_split,
            "pruned_total": total_pruned,
            "pruned_opacity": pruned_opacity,
            "pruned_screen": pruned_screen,
            "pruned_world": pruned_world,
        }

    # ---- Opacity reset (official L258-261) ----------------------------------

    def reset_opacity(self):
        """Global opacity reset: min(opacity, 0.01). Resets opacity moment rows.

        Official L258-261:
            opacities_new = inverse_sigmoid(min(get_opacity, 0.01))
            replace_tensor_to_optimizer(opacities_new, "opacity")
        """
        opacities_new = torch.logit(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )  # 1D
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # ---- Checkpoint ---------------------------------------------------------

    def capture(self) -> dict:
        """Save model state for checkpointing."""
        return {
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz.detach().cpu(),
            "shs": self._shs.detach().cpu(),
            "scaling": self._scaling.detach().cpu(),
            "rotation": self._rotation.detach().cpu(),
            "opacity": self._opacity.detach().cpu(),
            "max_radii2D": self.max_radii2D.detach().cpu(),
            "xyz_gradient_accum": self.xyz_gradient_accum.detach().cpu(),
            "denom": self.denom.detach().cpu(),
            "spatial_lr_scale": self.spatial_lr_scale,
            "num_points": self._xyz.shape[0],
        }

    def restore(self, state: dict, config: dict):
        """Restore from checkpoint."""
        self.active_sh_degree = state["active_sh_degree"]
        self.spatial_lr_scale = state["spatial_lr_scale"]
        self._xyz = nn.Parameter(state["xyz"].cuda().requires_grad_(True))
        self._shs = nn.Parameter(state["shs"].cuda().requires_grad_(True))
        self._scaling = nn.Parameter(state["scaling"].cuda().requires_grad_(True))
        self._rotation = nn.Parameter(state["rotation"].cuda().requires_grad_(True))
        self._opacity = nn.Parameter(state["opacity"].cuda().requires_grad_(True))
        self.max_radii2D = state["max_radii2D"].cuda()
        self.xyz_gradient_accum = state["xyz_gradient_accum"].cuda()
        self.denom = state["denom"].cuda()
        self.training_setup(config)

        # Restore explicit Gaussian IDs (R0.2 Part H)
        N = self._xyz.shape[0]
        self._gaussian_ids = torch.arange(N, dtype=torch.long, device="cuda")
        self._next_gaussian_id = N
