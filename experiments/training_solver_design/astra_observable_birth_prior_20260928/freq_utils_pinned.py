import torch
import torch.nn.functional as F
from PIL import ImageFilter
from gaussian_renderer import render_structgs
from .loss_utils import l1_loss, frequency_loss_simple,get_structure_tensor_torch
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms
from utils.general_utils import build_rotation
import random


def sampling_cameras(my_viewpoint_stack, mode="fps", num_cams=60, weights=None):
    ''' 
    Sample a given number of cameras from the viewpoint stack.
    
    Args:
        my_viewpoint_stack: List of camera viewpoints
        mode: "random" for random sampling, "fps" for farthest point sampling
    
    Returns:
        camlist: List of sampled cameras
    '''

    num_cams = num_cams
    
    if mode == "random":
        # Original random sampling
        camlist = []
        for _ in range(num_cams):
            loc = random.randint(0, len(my_viewpoint_stack) - 1)
            camlist.append(my_viewpoint_stack.pop(loc))
        return camlist
    
    elif mode == "fps":
        # Farthest Point Sampling based on camera positions
        num_cams = min(num_cams, len(my_viewpoint_stack))
        
        # Extract camera positions from world_view_transform
        # The camera position in world space is -R^T @ t where [R|t] is the view matrix
        camera_positions = []
        for cam in my_viewpoint_stack:
            # world_view_transform is [4, 4], extract rotation and translation
            w2c = cam.world_view_transform.transpose(0, 1)  # [4, 4]
            R = w2c[:3, :3]  # [3, 3]
            t = w2c[:3, 3]   # [3]
            # Camera position in world coordinates: -R^T @ t
            cam_pos = -R.T @ t
            camera_positions.append(cam_pos)
        
        camera_positions = torch.stack(camera_positions)  # [N, 3]
        
        # Farthest Point Sampling
        N = len(my_viewpoint_stack)
        selected_indices = []
        distances = torch.full((N,), float('inf'), device=camera_positions.device)
        
        # Start with a random camera
        current_idx = random.randint(0, N - 1)
        selected_indices.append(current_idx)
        
        for _ in range(num_cams - 1):
            # Update distances to the nearest selected point
            current_pos = camera_positions[current_idx]
            dists_to_current = torch.norm(camera_positions - current_pos, dim=1)
            distances = torch.minimum(distances, dists_to_current)
            
            # Exclude already selected points
            distances[selected_indices] = -1
            
            # Select the farthest point
            current_idx = torch.argmax(distances).item()
            selected_indices.append(current_idx)
        
        # Extract selected cameras and remove from stack
        camlist = []
        # Sort indices in descending order to avoid index shifting issues when popping
        for idx in sorted(selected_indices, reverse=True):
            camlist.append(my_viewpoint_stack.pop(idx))
        
        # Reverse to maintain original order
        camlist.reverse()
        
        return camlist
        
    
    else:
        raise ValueError(f"Unknown sampling mode: {mode}. Use 'random' or 'fps'.")




def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    return l1_loss_norm

def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss

def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def compute_projected_axes_subset(means2D, depths, scales, rotations, viewpoint_camera):
    """
    Computes projected 2D axes for a SUBSET of Gaussians using pre-computed means2D/depths.
    """
    with torch.no_grad():
        fx = viewpoint_camera.focal_x
        fy = viewpoint_camera.focal_y
        
        # 1. Back-project means2D to Camera Space (x, y)
        # u = (x/z)*fx + cx  =>  x = (u - cx) * z / fx
        # We need this to build the Jacobian at the correct location
        W = viewpoint_camera.image_width
        H = viewpoint_camera.image_height
        
        # Normalize means2D to pixels relative to center (assuming cx=W/2, cy=H/2 for simplicity 
        # or use actual projection matrix if needed, but standard pinhole approx is usually fine here)
        # Note: means2D from rasterizer is in Pixel Coordinates [0, W], [0, H]
        
        vec_x = (means2D[:, 0] - W * 0.5) * (depths / fx)
        vec_y = (means2D[:, 1] - H * 0.5) * (depths / fy)
        
        # 2. Build Jacobian J per point
        # J = [ fx/z   0     -fx*x/z^2 ]
        #     [ 0      fy/z  -fy*y/z^2 ]
        
        inv_z = 1.0 / (depths + 1e-7)
        inv_z2 = inv_z * inv_z
        
        J_00 = fx * inv_z
        J_02 = -fx * vec_x * inv_z2
        J_11 = fy * inv_z
        J_12 = -fy * vec_y * inv_z2
        
        # 3. Rotate and Scale Axes in Camera Space
        # Get World->View rotation
        W_view = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
        R_view = W_view[:3, :3] # [3, 3]
        
        # Convert quaternions to rotation matrices [N, 3, 3]
        R_local = build_rotation(rotations) 
        
        # R_total = R_view @ R_local
        # Expand R_view to match batch size
        R_view_batch = R_view.unsqueeze(0).expand(scales.shape[0], -1, -1)
        R_total = torch.bmm(R_view_batch, R_local) # [N, 3, 3]
        
        # Scale axes: Axis_vectors = R_total * Scales
        # scales is [N, 3]. We broadcast multiply columns.
        # This gives us the 3 axes (columns) in Camera coordinates
        axes_cam = R_total * scales.unsqueeze(1) # [N, 3, 3]
        
        # 4. Project Axes to 2D
        ax_x = axes_cam[:, 0, :] # [N, 3] (x-component of axis 0, 1, 2)
        ax_y = axes_cam[:, 1, :]
        ax_z = axes_cam[:, 2, :]
        
        # u = J00*x + J02*z
        # v = J11*y + J12*z
        u_vec = J_00.unsqueeze(1) * ax_x + J_02.unsqueeze(1) * ax_z
        v_vec = J_11.unsqueeze(1) * ax_y + J_12.unsqueeze(1) * ax_z
        
        # [N, 3, 2]
        return torch.stack([u_vec, v_vec], dim=2)


def update_freq_stats_online(viewpoint_cam, gaussians, cov2D, visibility_filter, structure_tensor_cache, viewspace_point_tensor=None, grad_threshold=None, transmittance_threshold=0.0, opacity_threshold=0.05, eta_compute_mode="wavelength"):
    """
    Online accumulation of frequency statistics for a single view.
    """
    if structure_tensor_cache is None:
        return

    # 1. Get Indices of Gaussians touching the view
    if visibility_filter.dtype == torch.bool:
        global_indices = torch.nonzero(visibility_filter).flatten()
    else:
        global_indices = visibility_filter
    
    if global_indices.shape[0] == 0:
        return

    # 2. Extract Data from Cov2D for visible points
    # Use integer indices for extraction
    current_cov2D = cov2D[global_indices] # [N_vis, 7]
    
    means2D_computed = current_cov2D[:, 3:5] # [N_vis, 2]
    depths = current_cov2D[:, 5]             # [N_vis]
    max_transmittance = current_cov2D[:, 6]  # [N_vis]

    # 3. Transmittance AND Opacity-based Active Mask
    current_opacities = gaussians.get_opacity[global_indices].squeeze(-1)
    is_active_vis = (max_transmittance > transmittance_threshold) & (current_opacities > opacity_threshold)
    
    # [NEW] Gradient-Based Masking
    if viewspace_point_tensor is not None and grad_threshold is not None:
        if viewspace_point_tensor.grad is not None:
            # viewspace_point_tensor is [N_total, 3]
            # Extract grads for visible points
            # Standard 3DGS accumulates norm of first 2 dimensions (x, y)
            current_grads = viewspace_point_tensor.grad[global_indices] # [N_vis, 3]
            grad_norms = torch.norm(current_grads[:, :2], dim=-1)
            
            is_grad_high = grad_norms > grad_threshold
            is_active_vis = is_active_vis & is_grad_high

    # [NEW] Densify Count Filtering - Skip GS that have been densified too many times
    # This prevents over-densification in already densified regions
    max_densify_count = 3  
    # if hasattr(gaussians, 'densify_count') and gaussians.densify_count.numel() > 0:
    #     current_densify_counts = gaussians.densify_count[global_indices]
    #     is_not_overdensified = current_densify_counts <= max_densify_count
    #     is_active_vis = is_active_vis & is_not_overdensified

    # Filter to get "Active & Visible" indices in the global array
    valid_indices_global = global_indices[is_active_vis] # [N_valid]
    
    if valid_indices_global.shape[0] > 0:
        st_map = structure_tensor_cache[viewpoint_cam.image_name]
        h, w = viewpoint_cam.image_height, viewpoint_cam.image_width

        # [NEW] Extract Transmittance Weights for the valid subset
        # We use this to scale the frequency violation
        weights_valid = torch.ones_like(max_transmittance)[is_active_vis] # [N_valid]

        # 4. Get Subsets of Data for Valid Points
        means2D_valid = means2D_computed[is_active_vis]
        depths_valid = depths[is_active_vis]
        
        scales_valid = gaussians.get_scaling[valid_indices_global]
        rots_valid = gaussians.get_rotation[valid_indices_global]

        # --- [MODIFICATION START] STOCHASTIC JITTER SAMPLING ---
        # Instead of sampling only at the center, we sample a random point 
        # within the 1-sigma ellipsoid of the Gaussian.
        
        # 1. Extract 2D Covariance Components
        # current_cov2D format: [cov_xx, cov_xy, cov_yy, mean_x, mean_y, ...]
        cov_xx = current_cov2D[is_active_vis, 0]
        cov_xy = current_cov2D[is_active_vis, 1]
        cov_yy = current_cov2D[is_active_vis, 2]
        
        # 2. Generate Random Jitter using Cholesky Decomposition
        # We sample from a 2D multivariate normal N(0, Sigma).
        # Sigma = L @ L.T, where L is the lower triangular Cholesky factor.
        # L = [[sqrt(xx), 0],
        #      [xy/sqrt(xx), sqrt(yy - xy^2/xx)]]
        
        eps1 = torch.randn_like(cov_xx)
        eps2 = torch.randn_like(cov_xx)
        
        # L11 = sqrt(cov_xx)
        L11 = torch.sqrt(torch.clamp(cov_xx, min=1e-6))
        # L21 = cov_xy / L11
        L21 = cov_xy / L11
        # L22 = sqrt(cov_yy - L21^2)
        L22 = torch.sqrt(torch.clamp(cov_yy - L21**2, min=1e-6))
        
        jitter_x = L11 * eps1
        jitter_y = L21 * eps1 + L22 * eps2
        
        # 4. Apply Jitter to Sampling Coordinates
        sample_x = means2D_valid[:, 0] + jitter_x
        sample_y = means2D_valid[:, 1] + jitter_y
        
        # Mirror Jitter if outside boundary
        # out_x = (sample_x < 0) | (sample_x >= w)
        # jitter_x[out_x] *= -1
        # sample_x = means2D_valid[:, 0] + jitter_x

        # out_y = (sample_y < 0) | (sample_y >= h)
        # jitter_y[out_y] *= -1
        # sample_y = means2D_valid[:, 1] + jitter_y

        # 5. Compute Grid using JITTERED coordinates
        norm_x = (sample_x / (w - 1)) * 2 - 1
        norm_y = (sample_y / (h - 1)) * 2 - 1
        grid_valid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0) 
        # --- [MODIFICATION END] ---
        
        # 6. Sample ST [3, N_valid] -> (Sxx, Sxy, Syy)
        sampled_S = F.grid_sample(st_map, grid_valid, align_corners=True).squeeze() #, padding_mode='border'
        if len(sampled_S.shape) == 1: sampled_S = sampled_S.unsqueeze(1)
        sampled_S = sampled_S.permute(1, 0) # [N_valid, 3]

        # 7. Compute 3-Channel Eta
        axes_2d = compute_projected_axes_subset(means2D_valid, depths_valid, scales_valid, rots_valid, viewpoint_cam)
        
        # Sxx = sampled_S[:, 0].unsqueeze(1)
        # Sxy = sampled_S[:, 1].unsqueeze(1)
        # Syy = sampled_S[:, 2].unsqueeze(1)
        
        # u = axes_2d[:, :, 0] 
        # v = axes_2d[:, :, 1] 
        
        # # Projection Frequency Violation
        # eta_3ch = Sxx * u**2 + 2 * Sxy * u * v + Syy * v**2 # [N_valid, 3]

        if eta_compute_mode == "wavelength":
            # --- [MODE 1] TRUE SIZE / WAVELENGTH ETA ---

            # 1. Extract Structure Tensor Components
            # sampled_S is [N, 3] -> (Sxx, Sxy, Syy)
            raw_Sxx = sampled_S[:, 0]
            raw_Sxy = sampled_S[:, 1]
            raw_Syy = sampled_S[:, 2]

            # 2. Compute Local Texture Wavelength (The "Speed Limit" for Size)
            # We compute the principal eigenvalue (Lambda1) of the Structure Tensor.
            # Lambda1 represents the maximum frequency energy (squared gradient).
            trace = raw_Sxx + raw_Syy
            det = raw_Sxx * raw_Syy - raw_Sxy**2
            delta = torch.sqrt(torch.clamp((trace/2)**2 - det, min=0.0))

            lambda1 = (trace / 2) + delta  # Max Eigenvalue (High Frequency Energy)

            # Convert Energy to Wavelength (Pixels)
            # Frequency f ~ sqrt(lambda). Wavelength w ~ 1/f.
            # We add epsilon to prevent division by zero in flat regions.
            wavelength_min = 1.0 / (torch.sqrt(lambda1) + 1e-5) 

            # 3. Compute Projected Gaussian Axis Lengths
            # axes_2d is [N, 3, 2] -> (N Gaussians, 3 Axes, u/v coordinates)
            u_vec = axes_2d[:, :, 0] 
            v_vec = axes_2d[:, :, 1] 

            # The physical length of the Gaussian's axis on the screen in pixels
            axis_lengths = torch.sqrt(u_vec**2 + v_vec**2 + 1e-8) # [N, 3]

            # 4. Compute Eta as a Ratio
            # Eta = (Gaussian Size) / (Texture Feature Size)
            # If Eta > 1.0, the Gaussian is larger than the feature -> Aliasing -> Split.
            # We expand wavelength_min to match the 3 axes [N, 1]
            eta_3ch = axis_lengths / (wavelength_min.unsqueeze(1))
        
        elif eta_compute_mode == "projection":
            # --- [MODE 2] DIRECTIONAL PROJECTION ETA ---
            # This formula properly projects the structure tensor onto each Gaussian axis direction.
            # For horizontal lines: Syy is high, Sxx ≈ 0
            # - Axes aligned with X (u component) → low eta
            # - Axes aligned with Y (v component) → high eta
            
            # Extract Structure Tensor Components
            # sampled_S is [N, 3] -> (Sxx, Sxy, Syy)
            Sxx = sampled_S[:, 0].unsqueeze(1)  # [N, 1]
            Sxy = sampled_S[:, 1].unsqueeze(1)  # [N, 1]
            Syy = sampled_S[:, 2].unsqueeze(1)  # [N, 1]
            
            # Get projected axis directions
            # axes_2d is [N, 3, 2] -> (N Gaussians, 3 Axes, u/v coordinates)
            u = axes_2d[:, :, 0]  # [N, 3] - X component of each axis
            v = axes_2d[:, :, 1]  # [N, 3] - Y component of each axis
            
            # Projection Frequency Violation:
            # eta = axis^T @ S @ axis = Sxx*u² + 2*Sxy*u*v + Syy*v²
            # This is the quadratic form that measures frequency energy along each axis direction.
            eta_3ch = Sxx * u**2 + 2 * Sxy * u * v + Syy * v**2  # [N, 3]

            eta_3ch = torch.sqrt(eta_3ch)
            
        else:
            raise ValueError(f"Unknown eta_compute_mode: {eta_compute_mode}")
        
        # [NEW] Weight by Transmittance (Importance Sampling)
        # If the Gaussian is transparent or occluded, we care less about its aliasing.
        eta_3ch = eta_3ch * weights_valid.unsqueeze(1)

        eta_total = eta_3ch.sum(dim=1)
        
        gaussians.accum_eta[valid_indices_global] += eta_total
        gaussians.accum_view_count[valid_indices_global] += 1.0
        gaussians.accum_weights_valid[valid_indices_global] += weights_valid
        
        # Accumulate 3-Channel Eta (Sum, for Leaky Average)
        # Note: We are now accumulating "Weighted" Eta.
        # gaussians.max_eta_3ch[valid_indices_global] += eta_3ch
        current_max = gaussians.max_eta_3ch[valid_indices_global]
        gaussians.max_eta_3ch[valid_indices_global] = torch.max(current_max, eta_3ch)
        
        # [NEW] Multiview Consistency: Track high/mid/low eta per GS
        TAU_HIGH = 1.0  # Frequency violation threshold
        TAU_LOW = 0.1   # Background/smooth threshold
        
        # Get max eta across 3 axes for classification
        eta_max_scalar = eta_3ch.max(dim=1).values  # [N_valid]
        
        # Classify each observation
        is_high = eta_max_scalar > TAU_HIGH
        is_low = eta_max_scalar <= TAU_LOW
        is_mid = ~is_high & ~is_low
        
        # Update counts
        gaussians.eta_high_count[valid_indices_global[is_high]] += 1.0
        gaussians.eta_mid_count[valid_indices_global[is_mid]] += 1.0
        gaussians.eta_low_count[valid_indices_global[is_low]] += 1.0
        
        # Accumulate sums for high and mid (for computing average later)
        if is_high.any():
            gaussians.eta_high_sum_3ch[valid_indices_global[is_high]] += eta_3ch[is_high]
        if is_mid.any():
            gaussians.eta_mid_sum_3ch[valid_indices_global[is_mid]] += eta_3ch[is_mid]




