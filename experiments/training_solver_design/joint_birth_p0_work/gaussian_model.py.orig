#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, identity_gate
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


try:
    from diff_gaussian_rasterization_structgs import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def modify_functions(self):
        old_opacities = self.get_opacity.clone()
        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = identity_gate
        self._opacity = self.opacity_activation(old_opacities)

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.filter_3D = torch.empty(0)
        self.tmp_radii = None
        
        # [NEW] Online Accumulation Tensors
        self.accum_eta = torch.empty(0)
        self.accum_view_count = torch.empty(0)
        self.max_eta_3ch = torch.empty(0)
        self.accum_weights_valid = torch.empty(0)
        self.densify_count = torch.empty(0)  # Track how many times GS has been normally densified
        
        # [NEW] Multiview Consistency Attributes
        self.eta_high_count = torch.empty(0)   # Count of views with high eta
        self.eta_high_sum_3ch = torch.empty(0) # Sum of eta_3ch for high eta views
        self.eta_mid_count = torch.empty(0)    # Count of views with mid eta
        self.eta_mid_sum_3ch = torch.empty(0)  # Sum of eta_3ch for mid eta views
        self.eta_low_count = torch.empty(0)    # Count of views with low eta

        self.optimizer = None
        self.shoptimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self, optimizer_type):
        if optimizer_type == "default":
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict(),
            self.spatial_lr_scale,
        )
        else:
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum,
        xyz_gradient_accum_abs, 
        denom,
        opt_dict, 
        shopt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # [NEW] Online Accumulation Tensors
        self.accum_eta = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.accum_view_count = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_eta_3ch = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.accum_weights_valid = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.densify_count = torch.zeros((self.get_xyz.shape[0]), dtype=torch.int32, device="cuda")
        
        # [NEW] Multiview Consistency Attributes
        self.eta_high_count = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.eta_high_sum_3ch = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.eta_mid_count = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.eta_mid_sum_3ch = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.eta_low_count = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]

        adam_eps = 10**(-training_args.adam_eps_order)
        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=adam_eps, betas=(0.9, 0.999))
            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=adam_eps, betas=(0.9, 0.999),
                                               weight_decay=0)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=adam_eps)
        elif self.optimizer_type == "hybrid":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=adam_eps, betas=(0.9, 0.999),amsgrad=True)
            self.shoptimizer = SparseGaussianAdam(sh_l, lr=0.0, eps=adam_eps)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.scale_rotation_scheduler = training_args.scale_rotation_scheduler
        if self.scale_rotation_scheduler:
            self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr*3.0,
                                                            lr_final=training_args.scaling_lr * 0.5,
                                                            lr_delay_mult=training_args.position_lr_delay_mult,
                                                            max_steps=training_args.position_lr_max_steps)
            self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr*3.0,
                                                             lr_final=training_args.rotation_lr * 0.5,
                                                             lr_delay_mult=training_args.position_lr_delay_mult,
                                                             max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        xyz_lr = None
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                xyz_lr = lr
            elif self.scale_rotation_scheduler and param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group['lr'] = lr
            elif self.scale_rotation_scheduler and param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group['lr'] = lr
        return xyz_lr

    def optimizer_step(self, iteration):
        ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        if iteration <= 15000:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
            self.shoptimizer.step()
            self.shoptimizer.zero_grad(set_to_none = True)
            # if iteration % 1 == 0: #16
                
        elif iteration <= 20000:
            if iteration % 32 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        else:
            if iteration % 64 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('filter_3D')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        filter_3D = self.filter_3D.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, decay_factor=0.1):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        # opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.1)
        opacities_new = current_opacity_with_filter * decay_factor
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = self.inverse_opacity_activation(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        try:
            filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]
        except ValueError:
            filter_3D = np.zeros((xyz.shape[0], 1))

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                # Handle amsgrad state if present
                if "max_exp_avg_sq" in stored_state:
                    stored_state["max_exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    # Handle amsgrad state if present
                    if "max_exp_avg_sq" in stored_state:
                        stored_state["max_exp_avg_sq"] = stored_state["max_exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]
        
        if self.filter_3D is not None and self.filter_3D.numel() > 0:
            self.filter_3D = self.filter_3D[valid_points_mask]
            
        # [NEW] Prune Accumulators
        if self.accum_eta.numel() > 0:
            self.accum_eta = self.accum_eta[valid_points_mask]
            self.accum_view_count = self.accum_view_count[valid_points_mask]
            self.max_eta_3ch = self.max_eta_3ch[valid_points_mask]
            self.accum_weights_valid = self.accum_weights_valid[valid_points_mask]
        if self.densify_count.numel() > 0:
            self.densify_count = self.densify_count[valid_points_mask]
        # [NEW] Prune Multiview Consistency Attributes
        if self.eta_high_count.numel() > 0:
            self.eta_high_count = self.eta_high_count[valid_points_mask]
            self.eta_high_sum_3ch = self.eta_high_sum_3ch[valid_points_mask]
            self.eta_mid_count = self.eta_mid_count[valid_points_mask]
            self.eta_mid_sum_3ch = self.eta_mid_sum_3ch[valid_points_mask]
            self.eta_low_count = self.eta_low_count[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                    # Handle amsgrad state if present
                    if "max_exp_avg_sq" in stored_state:
                        stored_state["max_exp_avg_sq"] = torch.cat((stored_state["max_exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_filter_3D):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.tmp_radii is None:
            self.tmp_radii = torch.zeros((self.get_xyz.shape[0] - new_tmp_radii.shape[0]), device="cuda")
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # abs
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if self.filter_3D is not None and self.filter_3D.numel() > 0:
            self.filter_3D = torch.cat((self.filter_3D, new_filter_3D), dim=0)

        # [NEW] Resize Accumulators (Append zeros for new points)
        # Note: We append zeros because new points haven't been seen yet.
        n_new = new_xyz.shape[0]
        self.accum_eta = torch.cat((self.accum_eta, torch.zeros(n_new, device="cuda")))
        self.accum_view_count = torch.cat((self.accum_view_count, torch.zeros(n_new, device="cuda")))
        self.max_eta_3ch = torch.cat((self.max_eta_3ch, torch.zeros((n_new, 3), device="cuda")))
        self.accum_weights_valid = torch.cat((self.accum_weights_valid, torch.zeros(n_new, device="cuda")))
        # Note: densify_count appends zeros, caller may overwrite with inherited values
        self.densify_count = torch.cat((self.densify_count, torch.zeros(n_new, dtype=torch.int32, device="cuda")))
        # [NEW] Resize Multiview Consistency Attributes
        self.eta_high_count = torch.cat((self.eta_high_count, torch.zeros(n_new, device="cuda")))
        self.eta_high_sum_3ch = torch.cat((self.eta_high_sum_3ch, torch.zeros((n_new, 3), device="cuda")))
        self.eta_mid_count = torch.cat((self.eta_mid_count, torch.zeros(n_new, device="cuda")))
        self.eta_mid_sum_3ch = torch.cat((self.eta_mid_sum_3ch, torch.zeros((n_new, 3), device="cuda")))
        self.eta_low_count = torch.cat((self.eta_low_count, torch.zeros(n_new, device="cuda")))

    def densify_and_split_structgs(self, metric_mask, max_eta_3ch=None, scale_power=1.0):
        """
        Splits Gaussians with Analytic Anisotropic 3-Channel Guidance.
        UPDATED: Computes kx, ky, kz separately based on sampling theory.
        k ~ sqrt(eta) reflects the sampling rate required to resolve the frequency violation.
        
        Args:
            scale_power: Exponent for scale division (new_scale = old_scale / k^scale_power).
                         1.0 = linear division, 2.0 = more aggressive shrinking.
        """
        new_xyz_list = []
        new_f_dc_list = []
        new_f_rest_list = []
        new_opacity_list = []
        new_scaling_list = []
        new_rotation_list = []
        new_radii_list = []
        new_filter_3D_list = []
        
        # [NEW] Accumulators for new stats
        new_accum_eta_list = []
        new_accum_view_count_list = []
        new_max_eta_3ch_list = []
        new_densify_count_list = []  # Inherit parent's count + 1

        n_total = self.get_xyz.shape[0]
        
        # Handle Padding
        if metric_mask.shape[0] < n_total:
            padding = torch.zeros(n_total - metric_mask.shape[0], dtype=torch.bool, device="cuda")
            metric_mask = torch.cat((metric_mask, padding))
            
        if max_eta_3ch is not None and max_eta_3ch.shape[0] < n_total:
            padding = torch.zeros((n_total - max_eta_3ch.shape[0], 3), device="cuda")
            max_eta_3ch = torch.cat((max_eta_3ch, padding))
            
        # Indices of points to split
        split_indices = torch.nonzero(metric_mask).squeeze(1)
        # Handle single-point case: ensure split_indices is always 1D
        if split_indices.dim() == 0:
            split_indices = split_indices.unsqueeze(0)
        if split_indices.numel() == 0:
            return

        # Prepare parameters for split candidates
        current_scales = self.get_scaling[split_indices]
        current_rots = self._rotation[split_indices]
        current_xyz = self.get_xyz[split_indices]
        
        # --- Determine k (splits per axis) ---
        if max_eta_3ch is not None:
            # Frequency-based splitting
            etavals = max_eta_3ch[split_indices] # [K, 3]
            
            # Sampling Theory:
            # eta is the frequency energy ratio (~ (sigma * omega_max)^2 ).
            # To resolve the aliasing, we need to reduce sigma by factor k such that sigma_new * omega_max <= 1.
            # sigma_new = sigma / k => (sigma/k)^2 * omega^2 <= 1 => eta / k^2 <= 1 => k >= sqrt(eta).
            
            # We calculate k separately for each axis.
            # Clamp min=2 (no split) and max=8 (limit VRAM usage).
            ks = torch.sqrt(torch.clamp(etavals, min=1.0)).ceil().int()
            # ks = etavals.ceil().int()
            ks = torch.clamp(ks, min=1)
            
        else:
            # Gradient-based split fallback (Standard 3DGS behavior)
            # Splits the longest axis by a factor of 2 (creates 2 gaussians)
            # Here we simulate that by setting k=2 on the max axis.
            max_scale_vals, max_scale_indices = torch.max(current_scales, dim=1)
            ks = torch.ones((current_scales.shape[0], 3), dtype=torch.int, device="cuda")
            ks.scatter_(1, max_scale_indices.unsqueeze(1), 2)
        
        # Group by configuration of (kx, ky, kz)
        # --- Analytic Split Logic (Vectorized) ---
        # Total splits N = kx * ky * kz
        N_per_point = ks.prod(dim=1) # [M]
        
        # Repeat parent attributes for each child
        # [Sum(N), ...]
        repeats = N_per_point
        
        # 1. Expand Parent Attributes
        # new_scaling = self.scaling_inverse_activation(current_scales / ks.float()).repeat_interleave(repeats, dim=0)
        new_scaling = self.scaling_inverse_activation(current_scales / ks.float()**scale_power).repeat_interleave(repeats, dim=0)
        new_rotation = current_rots.repeat_interleave(repeats, dim=0)
        new_features_dc = self._features_dc[split_indices].repeat_interleave(repeats, dim=0)
        new_features_rest = self._features_rest[split_indices].repeat_interleave(repeats, dim=0)
        new_opacity = self._opacity[split_indices].repeat_interleave(repeats, dim=0)
        new_radii = self.tmp_radii[split_indices].repeat_interleave(repeats, dim=0)
        new_filter_3D = self.filter_3D[split_indices].repeat_interleave(repeats, dim=0)
        new_densify_count = (self.densify_count[split_indices] + 1).repeat_interleave(repeats)

        # 2. Generate Grid Coordinates
        # We need to generate indices (ix, iy, iz) for each child j of parent p
        # where 0 <= ix < kx_p, etc.
        
        # create a flat range of indices [0, 1, ..., Sum(N)-1]
        total_children = repeats.sum()
        # computes start index for each parent in the flattened array
        # starts[i] = sum(N[:i])
        starts = torch.cumsum(repeats, dim=0) - repeats
        
        # expand starts to match children: [0, 0, ..., 0, 1, 1, ..., 1] 
        # (but we want the start index value)
        starts_expanded = starts.repeat_interleave(repeats)
        
        # local_index = global_index - start_index_of_parent
        child_indices = torch.arange(total_children, device="cuda") - starts_expanded
        
        # Retrieve repeated k values for modulo operations
        ks_repeated = ks.repeat_interleave(repeats, dim=0) # [Sum(N), 3]
        kx_r = ks_repeated[:, 0]
        ky_r = ks_repeated[:, 1]
        kz_r = ks_repeated[:, 2]
        
        # Decode (ix, iy, iz) from child_indices
        # index = ix * (ky*kz) + iy * kz + iz 
        # This follows the meshgrid order (indexing='ij') where Z varies fastest
        iz = child_indices % kz_r
        iy = (child_indices // kz_r) % ky_r
        ix = child_indices // (ky_r * kz_r)
        
        # Convert integer indices to centered coordinates
        # coord = i - (k-1)/2.0
        grid_x = ix.float() - (kx_r.float() - 1.0) / 2.0
        grid_y = iy.float() - (ky_r.float() - 1.0) / 2.0
        grid_z = iz.float() - (kz_r.float() - 1.0) / 2.0
        
        grid_flat = torch.stack([grid_x, grid_y, grid_z], dim=1) # [Sum(N), 3]
        
        # 3. Compute Offsets
        # divisions = sigma_new * sqrt(12)
        # sigma_new = sigma_old / k
        # We already computed new scales (pre-activation), but we need the raw sigma_new for offset calc.
        # current_scales is [M, 3]. 
        scales_repeated = current_scales.repeat_interleave(repeats, dim=0)
        stds_new_repeated = scales_repeated / ks_repeated.float()
        separations = stds_new_repeated * (12**0.5)
        
        local_offsets = separations * grid_flat # [Sum(N), 3]
        
        # 4. Rotate Offsets
        # rots_sub = current_rots.repeat_interleave(repeats, dim=0) -> new_rotation (already computed)
        R_sub = build_rotation(new_rotation)
        
        # bmm needs [B, 3, 3] x [B, 3, 1] -> [B, 3, 1]
        # local_offsets.unsqueeze(-1) is [Sum(N), 3, 1]
        world_offsets = torch.bmm(R_sub, local_offsets.unsqueeze(-1)).squeeze(-1)
        
        new_xyz = current_xyz.repeat_interleave(repeats, dim=0) + world_offsets

        # 5. Append to lists (now just single tensors)
        new_xyz_list.append(new_xyz)
        new_scaling_list.append(new_scaling)
        new_rotation_list.append(new_rotation)
        new_f_dc_list.append(new_features_dc)
        new_f_rest_list.append(new_features_rest)
        new_opacity_list.append(new_opacity)
        new_radii_list.append(new_radii)
        new_filter_3D_list.append(new_filter_3D)
        new_densify_count_list.append(new_densify_count)
        
        # Accumulators (zeros)
        n_new_total_sub = new_xyz.shape[0]
        new_accum_eta_list.append(torch.zeros(n_new_total_sub, device="cuda"))
        new_accum_view_count_list.append(torch.zeros(n_new_total_sub, device="cuda"))
        new_max_eta_3ch_list.append(torch.zeros((n_new_total_sub, 3), device="cuda"))

        # Prune Original Points
        total_prune_mask = torch.zeros(n_total, dtype=torch.bool, device="cuda")
        total_prune_mask[split_indices] = True
        
        if len(new_xyz_list) > 0:
            self.densification_postfix(
                torch.cat(new_xyz_list),
                torch.cat(new_f_dc_list),
                torch.cat(new_f_rest_list),
                torch.cat(new_opacity_list),
                torch.cat(new_scaling_list),
                torch.cat(new_rotation_list),
                torch.cat(new_radii_list),
                torch.cat(new_filter_3D_list)
            )

            n_new_total = self.get_xyz.shape[0]
            n_added = n_new_total - n_total
            
            # [NEW] Update densify_count for new children (inherited count + 1)
            # densification_postfix appends zeros, we overwrite with inherited values
            self.densify_count = torch.cat((self.densify_count[:n_total], torch.cat(new_densify_count_list)))
            
            full_prune_filter = torch.cat((total_prune_mask, torch.zeros(n_added, device="cuda", dtype=bool)))
            self.prune_points(full_prune_filter)

    def expand_undersized_gs(self, tau_expand, max_eta_3ch):
        """
        Analytically expands undersized Gaussians to the exact correct scale.
        
        Theory:
        - eta = (sigma * omega)^2 where sigma is scale, omega is max spatial frequency
        - To satisfy Nyquist (eta = 1): sigma_target = sigma_old / sqrt(eta)
        - In log-space: log(sigma_target) = log(sigma_old) - 0.5 * log(eta)
        """
        if max_eta_3ch is None:
            return

        # Identify undersized axes (eta < tau_expand and eta > 0)
        undersized_mask = (max_eta_3ch < tau_expand) & (max_eta_3ch > 0)
        
        if not undersized_mask.any():
            return

        with torch.no_grad():
            # Direct analytical update: log_scale_new = log_scale_old - 0.5 * log(eta)
            # This makes eta_new = 1.0 exactly
            eta_vals = torch.clamp(max_eta_3ch[undersized_mask], min=1e-6)
            delta_log_scale = -0.5 * torch.log(eta_vals)  # Direct correction
            
            delta = torch.zeros_like(self._scaling)
            delta[undersized_mask] = delta_log_scale
            
            new_scaling = self._scaling + delta
            
            # Replace in optimizer
            optimizable_tensors = self.replace_tensor_to_optimizer(new_scaling, "scaling")
            self._scaling = optimizable_tensors["scaling"]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_filter_3D = self.filter_3D[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_filter_3D)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_filter_3D = self.filter_3D[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_filter_3D)

    def densify_and_prune(self, max_grad,max_grad_abs, min_opacity, extent, max_screen_size, radii):
        self.tmp_radii = radii

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads_abs, max_grad_abs, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def densify_and_clone_structgs(self, metric_mask, filter):
        selected_pts_mask = torch.logical_and(metric_mask, filter)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_filter_3D = self.filter_3D[selected_pts_mask]
        
        # [NEW] Clone inherits parent's densify_count (no increment)
        parent_counts = self.densify_count[selected_pts_mask]
        n_old = self.get_xyz.shape[0]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_filter_3D)
        
        # Overwrite the zeros appended by densification_postfix with inherited counts
        n_new = new_xyz.shape[0]
        self.densify_count[n_old:n_old+n_new] = parent_counts


    def densify_and_prune_structgs(self, max_screen_size, min_opacity, extent, radii, args, 
                                 importance_score=None, pruning_score=None, 
                                 custom_split_mask=None, custom_prune_mask=None, 
                                 viewspace_points_indices=None,
                                 max_eta_3ch=None 
                                 ): 
        
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= args.grad_thresh, True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh, True, False)
        
        # --- MERGING RULES ---
        full_split_mask = torch.zeros_like(grad_qualifiers, dtype=torch.bool)
        full_prune_mask = torch.zeros_like(grad_qualifiers, dtype=torch.bool)
        
        if custom_split_mask is not None:
            if viewspace_points_indices is not None:
                full_split_mask[viewspace_points_indices] = custom_split_mask
            else:
                full_split_mask = custom_split_mask
            
        if custom_prune_mask is not None:
            if viewspace_points_indices is not None:
                full_prune_mask[viewspace_points_indices] = custom_prune_mask
            else:
                full_prune_mask = custom_prune_mask

        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense*extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense*extent

        final_split_mask = torch.logical_and(
            torch.logical_or(full_split_mask, grad_qualifiers_abs), 
            split_qualifiers
        )
        
        final_clone_mask = torch.logical_and(
            torch.logical_or(full_split_mask, grad_qualifiers), 
            # grad_qualifiers,
            clone_qualifiers
        )

        # 4. Execute Split/Clone
        metric_mask = importance_score > args.importance_score_threshold if importance_score is not None else torch.ones_like(final_clone_mask)

        self.densify_and_clone_structgs(metric_mask, final_clone_mask)
        
        # Call Analytic Split
        combined_split_mask = torch.logical_and(metric_mask, final_split_mask)
        self.densify_and_split_structgs(combined_split_mask, max_eta_3ch=max_eta_3ch, scale_power=args.ks_scale_power)

        # --- Pruning ---
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # Apply custom prune mask
        n_new = prune_mask.shape[0]
        n_old = full_prune_mask.shape[0]
        if n_new > n_old:
            padding = torch.zeros(n_new - n_old, dtype=torch.bool, device="cuda")
            full_prune_mask = torch.cat([full_prune_mask, padding])
            
        prune_mask = torch.logical_or(prune_mask, full_prune_mask)

        if pruning_score is not None:
            scores = 1 - pruning_score 
            to_remove = torch.sum(prune_mask)
            remove_budget = int(0.5 * to_remove)

            if remove_budget:
                n_init_points = self.get_xyz.shape[0]
                padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
                padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
                selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
                sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
                selected_pts_mask[sampled_indices] = True
                final_prune = torch.logical_and(prune_mask, selected_pts_mask)
                self.prune_points(final_prune)
        else:
            self.prune_points(prune_mask)
        
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()
        
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def final_prune_structgs(self, min_opacity, pruning_score = None):
        """Final-stage pruning: remove Gaussians based on opacity and multi-view consistency.
        In the final stage we remove Gaussians that have low opacity or that are flagged by
        our multi-view reconstruction consistency metric (provided as `pruning_score`)."""
        prune_mask = (self.get_opacity < min_opacity).squeeze() 
        scores_mask = pruning_score > 0.9
        final_prune = torch.logical_or(prune_mask, scores_mask)
        self.prune_points(final_prune)
