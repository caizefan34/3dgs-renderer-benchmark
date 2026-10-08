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
import os, random, time
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss,l2_loss, get_multiscale_structure_tensor_v1,get_multiscale_structure_tensor_v2
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_structgs, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.sh_utils import RGB2SH
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.freq_utils import sampling_cameras, update_freq_stats_online
import torch.nn.functional as F

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets, ply_path, prune_iterations):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    if ply_path:
        gaussians.load_ply(ply_path)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if ply_path:
        gaussians.spatial_lr_scale = scene.cameras_extent
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # [NEW] Camera ID mapping for logging
    cam_to_idx = {cam: i for i, cam in enumerate(scene.getTrainCameras())}


    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    ema_rgb_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    img_num = -1

    highresolution_index = []
    for index, camera in enumerate(scene.getTrainCameras()):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    if opt.sample_bbox_faces:
        # [NEW] Sample new GS to cover the 6 faces of bounding box at a fixed resolution
        start = time.time()
        with torch.no_grad():
            print("Sampling 6 faces of bounding box...")
            xyz = gaussians.get_xyz
            min_bound = xyz.min(dim=0)[0]
            max_bound = xyz.max(dim=0)[0]
            
            N = 16 # Fixed resolution
            
            new_xyz_list = []
            
            # X-faces
            y_range = torch.linspace(min_bound[1], max_bound[1], N, device="cuda")
            z_range = torch.linspace(min_bound[2], max_bound[2], N, device="cuda")
            Y, Z = torch.meshgrid(y_range, z_range, indexing='ij')
            Y = Y.reshape(-1)
            Z = Z.reshape(-1)
            
            X_min = torch.full_like(Y, min_bound[0])
            new_xyz_list.append(torch.stack([X_min, Y, Z], dim=-1))
            
            X_max = torch.full_like(Y, max_bound[0])
            new_xyz_list.append(torch.stack([X_max, Y, Z], dim=-1))
            
            # Y-faces
            x_range = torch.linspace(min_bound[0], max_bound[0], N, device="cuda")
            X, Z = torch.meshgrid(x_range, z_range, indexing='ij')
            X = X.reshape(-1)
            Z = Z.reshape(-1)
            
            Y_min = torch.full_like(X, min_bound[1])
            new_xyz_list.append(torch.stack([X, Y_min, Z], dim=-1))
            
            Y_max = torch.full_like(X, max_bound[1])
            new_xyz_list.append(torch.stack([X, Y_max, Z], dim=-1))
            
            # Z-faces
            X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
            X = X.reshape(-1)
            Y = Y.reshape(-1)
            
            Z_min = torch.full_like(X, min_bound[2])
            new_xyz_list.append(torch.stack([X, Y, Z_min], dim=-1))
            
            Z_max = torch.full_like(X, max_bound[2])
            new_xyz_list.append(torch.stack([X, Y, Z_max], dim=-1))
            
            all_new_xyz = torch.cat(new_xyz_list, dim=0)
            num_new = all_new_xyz.shape[0]
            
            grey = torch.tensor([0.5, 0.5, 0.5], device="cuda").unsqueeze(0).repeat(num_new, 1)
            features_dc = RGB2SH(grey).unsqueeze(1)
            features_rest = torch.zeros((num_new, (gaussians.max_sh_degree + 1) ** 2 - 1, 3), device="cuda")
            
            opacities = gaussians.inverse_opacity_activation(0.05 * torch.ones((num_new, 1), device="cuda"))
            
            step = (max_bound - min_bound).mean() / N
            dist2 = step ** 2
            scales = torch.log(torch.sqrt(dist2)).unsqueeze(0).repeat(num_new, 3) 
            
            rots = torch.zeros((num_new, 4), device="cuda")
            rots[:, 0] = 1
            
            new_tmp_radii = torch.zeros((num_new), device="cuda")
            new_filter_3D = torch.zeros((num_new, 1), device="cuda")
            
            gaussians.densification_postfix(all_new_xyz, features_dc, features_rest, opacities, scales, rots, new_tmp_radii, new_filter_3D)
            print(f"Added {num_new} points on bounding box faces.")
            print(f"Sampling 6 faces of bounding box took {time.time() - start:.4f} seconds")

    if opt.compute_3d_filter:
        gaussians.compute_3D_filter(cameras=scene.getTrainCameras())
    else:
        # set to 0
        gaussians.filter_3D = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")

    print("Pre-computing Structure Tensors (Vectorized)...")
    structure_tensor_cache = {}
    import torchvision
    
    start = time.time()
    st_vis_dir = os.path.join(dataset.model_path, "structure_tensors")
    os.makedirs(st_vis_dir, exist_ok=True)

    # Group cameras by resolution
    cameras_by_res = {}
    for cam in scene.getTrainCameras():
        res = (cam.image_height, cam.image_width)
        if res not in cameras_by_res:
            cameras_by_res[res] = []
        cameras_by_res[res].append(cam)

    # Process each resolution group in batches
    for res, cams in cameras_by_res.items():
        pixel_count = res[0] * res[1]
        batch_size = 100

        print(f"  Processing {len(cams)} views at {res[1]}x{res[0]} (Batch size: {batch_size})")
        for i in range(0, len(cams), batch_size):
            batch_cams = cams[i : i + batch_size]
            img_batch = torch.stack([cam.original_image.cuda() for cam in batch_cams], dim=0)
            
            with torch.no_grad():
                if opt.st_mode == "v1":
                    st_batch = get_multiscale_structure_tensor_v1(img_batch, levels=opt.st_levels)
                elif opt.st_mode == "v2":
                    st_batch = get_multiscale_structure_tensor_v2(img_batch, levels=opt.st_levels)
                
                for j, cam in enumerate(batch_cams):
                    st_map = st_batch[j:j+1] # Keep (1, 3, H, W) shape
                    structure_tensor_cache[cam.image_name] = st_map
                    
                    # Normalize for visualization (Sxx, Sxy, Syy can be large, so we normalize per image)
                    # st_vis = st_map.clone().detach()
                    # st_vis = (st_vis - st_vis.min()) / (st_vis.max() - st_vis.min() + 1e-6)
                    # torchvision.utils.save_image(st_vis, os.path.join(st_vis_dir, f"{cam.image_name}.png"))
    print(f"Pre-computing Structure Tensors took {time.time() - start:.4f} seconds")

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_structgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        if iteration % 1 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if opt.camera_sampling == "fps":
                # [NEW] Farthest Point Sampling
                viewpoint_stack = sampling_cameras(viewpoint_stack, mode="fps", num_cams=len(viewpoint_stack))
            else:
                # Default Random Sampling
                random.shuffle(viewpoint_stack)
        
        # [NEW] Batch Training: accumulate gradients from multiple cameras
        batch_loss = 0.0
        batch_rgb_loss = 0.0
        last_radii = None
        last_visibility_filter = None
        
        for batch_idx in range(opt.batch_size):
            # Pop from the start of the FPS-sorted list
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                if opt.camera_sampling == "fps":
                    viewpoint_stack = sampling_cameras(viewpoint_stack, mode="fps", num_cams=len(viewpoint_stack))
                else:
                    random.shuffle(viewpoint_stack)
            viewpoint_cam = viewpoint_stack.pop(0)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            render_pkg = render_structgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            visibility_filter = visibility_filter.squeeze(1)
            cov2D = render_pkg["cov2D"] # [N, 7] Extract info for online accumulation

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            Ll2 = l2_loss(image, gt_image)
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)+ opt.lambda_l2 * Ll2

            # Combine with main loss, scale by batch size for proper gradient averaging
            loss = rgb_loss #/ opt.batch_size
            loss.backward()
            
            # Accumulate for logging
            batch_loss += loss.item() #* opt.batch_size
            batch_rgb_loss += rgb_loss.item()
            
            # Track last radii/visibility for optimizer step
            last_radii = radii
            last_visibility_filter = visibility_filter
            
            # [NEW] Online Accumulation (per camera in batch)
            with torch.no_grad():
                if iteration < opt.densify_until_iter and iteration % 10 == 0:
                    update_freq_stats_online(viewpoint_cam, gaussians, cov2D, visibility_filter, structure_tensor_cache, viewspace_point_tensor=viewspace_point_tensor, grad_threshold= opt.freq_grad_threshold, transmittance_threshold=opt.freq_transmittance_threshold, opacity_threshold=opt.freq_opacity_threshold, eta_compute_mode=opt.eta_compute_mode)
        
        # Use last camera for logging
        radii = last_radii
        visibility_filter = last_visibility_filter

        
        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Progress bar (use averaged batch loss)
            ema_loss_for_log = 0.4 * (batch_loss / opt.batch_size) + 0.6 * ema_loss_for_log

            gb_unit = 1024 ** 3
            alloc = torch.cuda.memory_allocated() / gb_unit
            rsrv = torch.cuda.memory_reserved() / gb_unit

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Num": f"{gaussians.get_xyz.shape[0]}",
                    "GPU Mem": f"{alloc:.2f} GB / {rsrv:.2f} GB"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_structgs, (pipe, background, opt.mult))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                # current total_time
                print("Current time: {}".format(total_time))
            
            optim_start.record()
            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                is_normal_densification = iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0
                is_warmup_densification = opt.warmup_densification and iteration > opt.densify_from_iter and iteration % 100 == 0 and not is_normal_densification
                if is_normal_densification:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    # 1. Calculate Gradients (Standard 3DGS metric)
                    grads = gaussians.xyz_gradient_accum / gaussians.denom
                    grads[grads.isnan()] = 0.0
                    
                    # 2. Create Gradient Mask 
                    # [CRITICAL] We MUST use this to prevent 7M points. 
                    # Only split if the geometry is struggling (high error).
                    is_grad_high = torch.norm(grads, dim=-1) >= 1e-5

                    # 3. [NEW] Multiview Consistency Criterion
                    # Compute ratios of high/low eta counts
                    valid_mask = gaussians.accum_view_count > 0
                    
                    high_ratio = torch.zeros_like(gaussians.accum_view_count)
                    low_ratio = torch.zeros_like(gaussians.accum_view_count)
                    high_ratio[valid_mask] = gaussians.eta_high_count[valid_mask] / gaussians.accum_view_count[valid_mask]
                    low_ratio[valid_mask] = gaussians.eta_low_count[valid_mask] / gaussians.accum_view_count[valid_mask]
                    
                    # 4. Split if consistently high eta across views 
                    split_mask = (high_ratio > opt.split_ratio_threshold) & is_grad_high
                    
                    # 5. Compute average high eta 3ch for densification guidance
                    avg_high_eta_3ch = torch.zeros_like(gaussians.max_eta_3ch)
                    has_high = gaussians.eta_high_count > 0
                    avg_high_eta_3ch[has_high] = gaussians.eta_high_sum_3ch[has_high] / gaussians.eta_high_count[has_high].unsqueeze(1)
                    max_high_eta = gaussians.max_eta_3ch
                    # 6. Prune if consistently low eta across views 
                    prune_mask = (low_ratio > opt.prune_ratio_threshold) & valid_mask

                    # [NEW] Expand undersized Gaussians
                    # gaussians.expand_undersized_gs(
                    #     tau_expand=opt.tau_expand,
                    #     max_eta_3ch=avg_high_eta_3ch
                    # )

                    # Pass the AVG high eta to the splitting function for analytic guidance
                    gaussians.densify_and_prune_structgs(
                        max_screen_size=size_threshold,
                        min_opacity=0.1, # 0.005 is a good default
                        extent=scene.cameras_extent,
                        radii=radii,
                        args=opt,
                        importance_score=gaussians.accum_view_count,
                        pruning_score=None,
                        custom_split_mask=split_mask,
                        custom_prune_mask=prune_mask,
                        viewspace_points_indices=None,
                        max_eta_3ch=max_high_eta  # Use max high eta for shaping the split
                    )
                    
                    # Reset accumulators
                    gaussians.accum_eta.zero_()
                    gaussians.accum_view_count.zero_()
                    gaussians.max_eta_3ch.zero_()
                    gaussians.accum_weights_valid.zero_()
                    # Reset multiview consistency accumulators
                    gaussians.eta_high_count.zero_()
                    gaussians.eta_high_sum_3ch.zero_()
                    gaussians.eta_mid_count.zero_()
                    gaussians.eta_mid_sum_3ch.zero_()
                    gaussians.eta_low_count.zero_()

                elif is_warmup_densification:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_grad_abs_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                    # [NEW] Apply multiview consistency prune strategy to warmup phase
                    # valid_mask = gaussians.accum_view_count > 0
                    # low_ratio = torch.zeros_like(gaussians.accum_view_count)
                    # low_ratio[valid_mask] = gaussians.eta_low_count[valid_mask] / gaussians.accum_view_count[valid_mask]
                    
                    # PRUNE_RATIO_THRESHOLD = 0.8
                    # prune_mask = (low_ratio > PRUNE_RATIO_THRESHOLD) & valid_mask
                    # print("Warmup Prune points: ", prune_mask.sum())
                    # gaussians.prune_points(prune_mask)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity(opt.opacity_reset_decay)

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    if opt.compute_3d_filter:
                        gaussians.compute_3D_filter(cameras=scene.getTrainCameras())

           
            if iteration in prune_iterations:
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack)

                prune_mask = (gaussians.get_opacity < 0.1).squeeze()
                gaussians.prune_points(prune_mask)
                # _, pruning_score, _, _, _,_ = compute_gaussian_score_structgs(camlist, gaussians, pipe, bg, opt)                    
                # gaussians.final_prune_structgs(min_opacity = 0.1, pruning_score = pruning_score)
        
            # Optimization step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                elif opt.optimizer_type == "hybrid":
                    visible = radii > 0
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    gaussians.shoptimizer.step(visible, radii.shape[0])
                    gaussians.shoptimizer.zero_grad(set_to_none = True)

            # record time
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    # scene.save(iteration)
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    parser.add_argument("--ply_path", type=str, default=None)
    parser.add_argument("--prune_iterations", nargs="+", type=int, default=[4000,8000])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    #from 100 to 3k, every 100
    # args.save_iterations= [i for i in range(200, 7100, 200)]
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.websockets,
        args.ply_path,
        args.prune_iterations
    )

    # All done
    print("\nTraining complete.")
