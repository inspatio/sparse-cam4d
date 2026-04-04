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

import os
import random
from copy import deepcopy
import torch
from torch import nn
from utils.loss_utils import l1_loss, ssim, PerceptualLoss, l2_loss
from utils.general_utils import get_expon_lr_func
from gaussian_renderer import differentiable_render_with_hexplane
import sys
from scene import Scene, MixGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torchvision.utils import make_grid
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from utils.data_utils import generate_dataloader
from utils.log_utils import TimeLogger
import json
import torchvision
from multiprocessing.pool import ThreadPool
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import datetime
from loguru import logger
time_logger = TimeLogger()
time_logger.enabled(False)

def save_pose(path, train_differentiable_cameras, transformsfile):
    with torch.no_grad():
        assert os.path.exists(transformsfile)
        with open(transformsfile) as json_file:
            contents = json.load(json_file)
        if "frames" in contents:
            frames = contents["frames"]
        else:
            frames = contents

        tbar = tqdm(range(len(frames)))
        def update_cam_pose(frame):
            camera_id = int(frame['id'])
            camera_info = deepcopy(frame)
            differentiable_cam = train_differentiable_cameras.get_camera(camera_id)
            if differentiable_cam is not None:
                R = differentiable_cam.R.detach().cpu().numpy()
                T = differentiable_cam.T.detach().cpu().numpy()
                w2c = np.eye(4)
                w2c[:3,:3] = R.T
                w2c[:3, 3] = T
                c2w = np.linalg.inv(w2c)

                camera_info["width"] = differentiable_cam.width
                camera_info["height"] = differentiable_cam.height
                camera_info["transform_matrix"] = c2w.tolist()
                camera_info["position"] = c2w[:3, 3].tolist()
                camera_info["rotation"] = c2w[:3,:3].tolist()
                camera_info["fx"] = differentiable_cam.fl_x.item()
                camera_info["fy"] = differentiable_cam.fl_y.item()
                camera_info["cx"] = differentiable_cam.cx.item()
                camera_info["cy"] = differentiable_cam.cy.item()

            tbar.update(1)

            return camera_info
        with ThreadPool() as pool:
            cam_infos = pool.map(update_cam_pose, frames)
            pool.close()
            pool.join()

        cam_infos = [cam_info for cam_info in cam_infos if cam_info is not None]
        cam_infos = sorted(cam_infos, key = lambda x: int(x["id"]))
        tbar.close()
        logger.info(f"Save {len(cam_infos)} poses")
        with open(path, "w") as f:
            json.dump(cam_infos, f, indent=4)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, resume_training, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size,
             target_cam, target_ts_train, target_ts_test, hyper):

    #! Time Duration Normalization
    time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    #! Backup Config File
    cfg_bkp_path = os.path.join(args.model_path, 'configs.yaml')
    os.system(f'cp {args.config} {cfg_bkp_path}')

    #! Initialize Scene
    min_timestep = 1.0/dataset.Ntime/dataset.frame_ratio
    scene = Scene(dataset, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration, target_cam=target_cam,
                  target_time_train=[ts*min_timestep for ts in target_ts_train], target_time_test=[ts*min_timestep for ts in target_ts_test])
    #! Initialize 4D-Gaussians with Distortion Fields
    gaussians = MixGaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d,
                                 sh_degree_t=2 if pipe.eval_shfs_4d else 0, distortion_args=hyper, frame_ratio=dataset.frame_ratio, Ntime=dataset.Ntime)
    if hyper.train_distortion:
        gaussians.setup_distortion_fields(target_ts_train, piece_length=50)
    logger.info(f"Initialize {len(gaussians._distortion_list)} distortion fields.")
    #! initial gaussians from ply
    gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)
    gaussians.training_setup(opt, scene.getTrainDifferentiableCameras())
    #! set hexplane grid resolution
    xyz_max = gaussians._xyz.detach().cpu().numpy().max(axis=0)
    xyz_min = gaussians._xyz.detach().cpu().numpy().min(axis=0)
    expand_ratio = 2.0
    center = (xyz_max + xyz_min) / 2.0
    half_diag = (xyz_max - xyz_min) / 2.0
    expanded_half_diag = half_diag * expand_ratio   # expand aabb diag
    xyz_min = center - expanded_half_diag
    xyz_max = center + expanded_half_diag
    for distortion in gaussians._distortion_list:
        distortion.deformation_net.set_aabb(xyz_max,xyz_min)
        distortion.deformation_net.set_max_delta(
            max_dx = torch.tensor(scene.cameras_extent),
            max_ds = torch.tensor(scene.cameras_extent),
            max_dr = torch.tensor(1.0)
        )

    #! resume training from checkpoint
    if checkpoint:
        assert os.path.exists(checkpoint), f"Checkpoint file {checkpoint} does not exist"
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, scene.getTrainDifferentiableCameras(), iteration=first_iter, resume_training=resume_training)
        logger.info(f"Loaded trained model from {checkpoint} at iteration {first_iter}")
        torch.cuda.empty_cache()

    #! set background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    #! set event for timing
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    #! set ema for logging
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0
    ema_lpipsloss_for_log = 0.0
    ema_depthloss_for_log = 0.0
    ema_hexplaneloss_for_log = 0.0
    ema_poseloss_for_log = 0.0
    ema_tvloss_for_log = 0.0
    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') and key!='lambda_dssim']
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0

    #! set progress bar
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter = first_iter if first_iter > 0 else first_iter + 1
    logger.info(f"Training from {first_iter} iteration")

    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3,pipe.env_map_res, pipe.env_map_res),dtype=torch.float, device="cuda").requires_grad_(True))
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None
    gaussians.env_map = env_map

    training_dataset, testing_dataset, training_dataloader, sampler = generate_dataloader(
        dataset, scene, gaussians, batch_size,
        train_target_time=[ts*min_timestep for ts in target_ts_train], test_target_time=[ts*min_timestep for ts in target_ts_test])

    #! Init Loss Function and Loss Weight
    logger.info(f"Init Perceptual Loss with {dataset.lpips_net}")
    perceptual_loss = PerceptualLoss(dataset.lpips_net, device=f'cuda:0', spatial=dataset.lpips_spatial)
    flow_l1_weight = get_expon_lr_func(opt.flow_l1_weight_init, opt.flow_l1_weight_final, max_steps=opt.iterations)     # initialized but not used
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)  # initialized but not used
    if dataset.lpips_decay: # disabled by default setting
        lpips_weight_init = 0.2
        lpips_weight_final = 0.02
        lpips_weight = get_expon_lr_func(lpips_weight_init, lpips_weight_final, max_steps=opt.iterations)
        logger.info(f"Enable Llpips Weights Decay")

    #! Setup Pose Optimization
    training_differentiable_cams = scene.getTrainDifferentiableCameras()
    pose_path = os.path.join(scene.model_path, 'pose')
    os.makedirs(pose_path, exist_ok=True)
    save_pose(os.path.join(pose_path, 'pose_org.json'), training_differentiable_cams,
                os.path.join(dataset.source_path, dataset.transforms_file))

    #! Freeze Distortion Fields from the beginning
    for distortion in gaussians._distortion_list:
        distortion.requires_grad_(False)
    logger.info(f"Freeze total {len(gaussians._distortion_list)} Distortion Fields")

    epoch = 0
    iteration = first_iter
    #! Training Loop
    while iteration < opt.iterations + 1:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1

        for batch_data in training_dataloader:
            #! Pose Optimization Switcher
            if iteration == opt.cam_optim_from_iter+1:
                if opt.lambda_pose > 0:
                    logger.info("Enable Camera Pose Optimization")
                    for cid in training_differentiable_cams.cameras:
                        training_differentiable_cams.get_camera(cid).requires_grad_(True)
                else:
                    logger.info("Not Consider Camera Pose Optimization")
            if iteration == opt.cam_optim_until_iter+1 and opt.lambda_pose > 0:
                logger.info("Disable Camera Pose Optimization")
                for cid in training_differentiable_cams.cameras:
                    training_differentiable_cams.get_camera(cid).requires_grad_(False)

            #! Distortion Fields Optimization Switcher
            if iteration == hyper.distortion_optim_from_iter + 1:
                for distortion in gaussians._distortion_list:
                    distortion.requires_grad_(True)
                logger.info(f"Update total {len(gaussians._distortion_list)} Distortion Fields")
            elif iteration == opt.densify_until_iter:
                for distortion in gaussians._distortion_list:
                    distortion.requires_grad_(False)
                logger.info(f"Freeze total {len(gaussians._distortion_list)} Distortion Fields")

            iter_start.record()
            update_lr_dict = gaussians.update_learning_rate(iteration)
            if tb_writer:
                for key, value in update_lr_dict.items():
                    tb_writer.add_scalar(f"lr/{key}", value, iteration)

            #! Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()

            #! Debug Mode Switcher
            if (iteration - 1) == debug_from:
                pipe.debug = True

            #! Init Batch Variables
            batch_point_grad = []
            batch_visibility_filter = []
            batch_radii = []
            batch_t_grad = []

            time_logger.update('before_batch_iter')
            for batch_idx in range(batch_size):
                #! Load Data
                gt_image, viewpoint_cam, alpha_mask = batch_data[batch_idx]
                gt_image = gt_image.cuda()
                viewpoint_cam = viewpoint_cam.cuda()
                alpha_mask = alpha_mask.cuda()
                differentiable_cam = training_differentiable_cams.get_camera(viewpoint_cam.camera_id)
                time_logger.update(f"{batch_idx} - load data")

                #! Render with Distortion Fields
                render_pkg = differentiable_render_with_hexplane(
                    viewpoint_cam, differentiable_cam, gaussians, pipe, background,
                    apply_distortion=(iteration >= hyper.distortion_optim_from_iter), # apply distortion fields after warmup stage
                    time_logger=time_logger, batch_idx=batch_idx
                )
                #! Get Render Results
                image, viewspace_point_tensor, visibility_filter, radii = \
                    render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                time_logger.update(f"{batch_idx} - render")

                #! Init Loss
                loss = torch.tensor(0.0, device="cuda")

                #* RGB Loss *#
                Ll1 = Lssim = Llpips = torch.tensor(0.0, device="cuda")
                if viewpoint_cam.is_true_image:
                    Ll1 = (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image) # (3, h, w)
                    Lssim = opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                else:
                    if opt.lambda_lpips > 0:
                        if dataset.lpips_decay:
                            Llpips = lpips_weight(iteration) * perceptual_loss(image[None].float(), gt_image[None].float(), normalize=True, mask=alpha_mask).sum()/alpha_mask.sum()
                        else:
                            Llpips = opt.lambda_lpips * perceptual_loss(image[None].float(), gt_image[None].float(), normalize=True, mask=alpha_mask).sum()/alpha_mask.sum()
                        Llpips *= 0.5
                        Ll1 = 0.02 * (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image, mask=alpha_mask)
                    else:
                        Ll1 = (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image, mask=alpha_mask) # (3, h, w)
                        Lssim = opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=alpha_mask))
                loss += Ll1 + Lssim + Llpips
                time_logger.update(f"{batch_idx} - rgb loss")

                #* Distortion Field Loss *#
                tv_loss = torch.tensor(0.0).cuda()
                Lhexplane = torch.tensor(0.0).cuda()
                if opt.lambda_hexplane > 0 and len(gaussians._distortion_list) and iteration > hyper.distortion_optim_from_iter and not viewpoint_cam.is_true_image:
                    #* Hexplane TV Loss (Smoothness) *#
                    tv_loss, _ = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
                    #* Hexplane TV Loss (Canonical) *#
                    if hyper.zero_canonical_weight > 0:
                        dx, ds, dr, do = render_pkg["distort_dx"], render_pkg["distort_ds"], render_pkg["distort_dr"], render_pkg["distort_do"]
                        zero_tensor = torch.tensor(0.0, device="cuda")
                        Lhexplane += l2_loss(dx, torch.zeros_like(dx)) if dx is not None else zero_tensor
                        Lhexplane += l1_loss(ds, torch.zeros_like(ds)) if ds is not None else zero_tensor
                        Lhexplane += l1_loss(dr, torch.zeros_like(dr)) if dr is not None else zero_tensor
                        Lhexplane += l1_loss(do, torch.zeros_like(do)) if do is not None else zero_tensor
                        Lhexplane = hyper.zero_canonical_weight * Lhexplane
                loss += Lhexplane + tv_loss
                time_logger.update(f"{batch_idx} - distortion loss")

                #* Pose Optimization Regularization *#
                Lpose = torch.tensor(0.0).cuda()
                if opt.lambda_pose > 0:
                    if iteration > opt.cam_optim_from_iter and iteration <= opt.cam_optim_until_iter:
                        Lpose = opt.lambda_pose * (
                                    torch.abs(differentiable_cam.T - differentiable_cam.T_origin).mean() +
                                    torch.abs(differentiable_cam.q - differentiable_cam.q_origin).mean()
                                )
                loss += Lpose
                time_logger.update(f"{batch_idx} - pose loss")

                #* depth loss
                Ldepth = torch.tensor(0.0).cuda()
                if iteration > opt.densify_from_iter and opt.lambda_depth > 0 and viewpoint_cam.depth is not None:
                    gt_depth = viewpoint_cam.depth.cuda().squeeze()
                    depth = render_pkg["depth"].cuda().cuda().squeeze()
                    depth_mask = torch.logical_and(depth > 1e-10, gt_depth > 1e-10)
                    gt_depth = 1.0 / (gt_depth + 1e-6)
                    depth = 1.0 / (depth + 1e-6)
                    valid_d_gt = gt_depth[depth_mask]
                    mean = valid_d_gt.mean()
                    std = valid_d_gt.std() + 1e-6
                    gt_depth_norm = (gt_depth - mean) / std
                    depth_norm = (depth - mean) / std
                    diff = depth_norm[depth_mask] - gt_depth_norm[depth_mask]
                    Ldepth = depth_l1_weight(iteration) * diff.abs().mean()
                loss += Ldepth
                time_logger.update(f"{batch_idx} - depth loss")

                # flow loss - not used
                Lflow = torch.tensor(0.0).cuda()
                if opt.lambda_flow > 0 and viewpoint_cam.flow is not None:
                    render_flow = render_pkg["flow"].cuda() # (2,h,w)
                    gt_flow = viewpoint_cam.flow.cuda()
                    large_motion_threshold = 0.2
                    large_motion_mask = torch.norm(gt_flow, p=2, dim=-1) > large_motion_threshold
                    if large_motion_mask.sum() != 0:
                        render_flow = render_flow.permute(1,2,0) # (h,w,2)
                        Lflow = flow_l1_weight(iteration) * torch.norm((gt_flow - render_flow)[large_motion_mask], p=2, dim=-1).mean()
                loss += Lflow
                time_logger.update(f"{batch_idx} - flow loss")

                # motion loss - not used
                Lmotion = torch.tensor(0.0).cuda()
                if gaussians.gaussian_dim == 4 and opt.lambda_motion > 0:
                    if iteration <= opt.densify_from_iter or opt.lambda_rigid <= 0:
                        _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                    Lmotion = opt.lambda_motion * velocity.norm(p=2, dim=1).mean()
                loss += Lmotion
                time_logger.update(f"{batch_idx} - motion loss")

                # Gaussian LifeSpan Regularization: Longer, Better - not used
                Llifespan = torch.tensor(0.0).cuda()
                if opt.lambda_lifespan > 0:
                    lifespan = gaussians.get_lifespan()
                    Llifespan = opt.lambda_lifespan * l1_loss(1.0 / (lifespan+1e-6), torch.zeros_like(lifespan))
                loss += Llifespan
                time_logger.update(f"{batch_idx} - lifespan loss")

                #! Draw Grad from Real Image Only (for densification)
                if viewpoint_cam.is_true_image:
                    loss_real = (Ll1+Lssim+Llpips) / batch_size
                    viewspace_point_tensor_grad, t_grad = torch.autograd.grad(loss_real, [viewspace_point_tensor, gaussians._t], retain_graph=True)
                    batch_point_grad.append(torch.norm(viewspace_point_tensor_grad[:,:2], dim=-1))
                    batch_radii.append(radii)
                    batch_visibility_filter.append(visibility_filter)
                    if gaussians.gaussian_dim == 4 and t_grad is not None:
                        batch_t_grad.append(t_grad[:, 0].detach())
                time_logger.update(f"{batch_idx} - autograd")

                #! Batched Loss Backward
                loss = loss / batch_size
                loss.backward()

                time_logger.update(f"{batch_idx} - loss backward")

            #! Batched Grad Calculation
            if batch_size > 1 and len(batch_visibility_filter):
                visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii,1).max(1)[0]

                batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)

                if gaussians.gaussian_dim == 4:
                    #! Draw Grad from Real Image Only (for densification)
                    batch_t_grad = torch.stack(batch_t_grad, 1).sum(1)  # (N,)
                    batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_t_grad = batch_t_grad.unsqueeze(1)
            else:
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone().detach()
            time_logger.update(f"Batch Grad")

            iter_end.record()

            #! Setup loss dictionary
            loss_dict = {
                "Ll1": Ll1,
                "Lssim": Lssim,
                "Llpips": Llpips,
                "Lhexplane": Lhexplane,
                "Ltv": tv_loss,
                "Lpose": Lpose,
                "Ldepth": Ldepth,
            }

            with torch.no_grad():
                #! Calculate EMA Losses
                psnr_for_log = psnr(image, gt_image).mean().double()
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                ema_ssimloss_for_log = 0.4 * Lssim.item() + 0.6 * ema_ssimloss_for_log
                ema_lpipsloss_for_log = 0.4 * Llpips.item() + 0.6 * ema_lpipsloss_for_log
                ema_hexplaneloss_for_log = 0.4 * Lhexplane.item() + 0.6 * ema_hexplaneloss_for_log
                ema_tvloss_for_log = 0.4 * tv_loss.item() + 0.6 * ema_tvloss_for_log
                ema_poseloss_for_log = 0.4 * Lpose.item() + 0.6 * ema_poseloss_for_log
                ema_depthloss_for_log = 0.4 * Ldepth.item() + 0.6 * ema_depthloss_for_log

                if iteration % 10 == 0:
                    postfix = {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "N": f"{gaussians.get_xyz.shape[0]}",
                        "PSNR": f"{psnr_for_log:.{2}f}",
                        "Ll1": f"{ema_l1loss_for_log:.{4}f}",
                        "Llpips": f"{ema_lpipsloss_for_log:.{4}f}",
                    }
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                #! Log
                training_report(tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                testing_iterations, scene, gaussians, differentiable_render_with_hexplane, (pipe, background),
                                training_dataset, testing_dataset, perceptual_loss, loss_dict)
                time_logger.update(f"Log")
                #! Save
                if (iteration in saving_iterations) or (iteration == opt.cam_optim_until_iter):
                    logger.info(f"[ITER {iteration}] Saving Gaussians")
                    scene.save(gaussians, iteration)
                    if (iteration in saving_iterations and iteration < opt.cam_optim_until_iter):
                        save_pose(os.path.join(pose_path, f'pose_{iteration}.json'), training_differentiable_cams,
                                    os.path.join(dataset.source_path, dataset.transforms_file))
                    if (iteration == opt.cam_optim_until_iter):
                        save_pose(os.path.join(pose_path, f'pose_{opt.cam_optim_until_iter}_final.json'), training_differentiable_cams,
                                    os.path.join(dataset.source_path, dataset.transforms_file))
                time_logger.update(f"Save")
                #! Densification
                if (opt.densify_until_iter < 0 or iteration < opt.densify_until_iter) and (opt.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < opt.densify_until_num_points):
                    # if (opt.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < opt.densify_until_num_points):
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if batch_size == 1:
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                    else:
                        #! Draw Grad from Real Image Only
                        if len(batch_visibility_filter):
                            gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None, denom=len(batch_visibility_filter))

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.thresh_opa_prune, scene.cameras_extent, size_threshold, opt.densify_grad_t_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        logger.info(f"reset opacity")
                        gaussians.reset_opacity()
                time_logger.update(f"Densification")

            #! Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if pipe.env_map_res and iteration < pipe.env_optimize_until:
                    env_map_optimizer.step()
                    env_map_optimizer.zero_grad(set_to_none = True)
                time_logger.update(f"Optimize")

            #! Efficiency Analysis
            if iteration % 500 == 0:
                time_logger.show()
            time_logger.reset()

            #! update iteration
            iteration += 1
            if iteration > opt.iterations:
                break

    logger.info(f"[ITER {opt.iterations}] Saving Gaussians")
    scene.save(gaussians, opt.iterations)
    if tb_writer:
        tb_writer.close()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    logger.info("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        # Set up tensorboard folder
        tb_folder = os.path.join(args.model_path, 'runs')
        assert len(args.experiment_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(tb_folder, f"{args.experiment_name}_{timestamp}")
        tb_writer = SummaryWriter(log_dir=log_dir)
    else:
        logger.warning("Tensorboard not available: not logging progress")
    return tb_writer

def prepare_logger(model_path):
    # Set up logger
    log_foldername = os.path.join(model_path, 'log')
    os.makedirs(log_foldername, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_foldername, f"log_{timestamp}.log")
    logger.info(f"Training Log save to {log_filename}")

    logger.add(log_filename, level="DEBUG", backtrace=True, diagnose=True)
    # 将未捕获异常输出到日志文件
    def log_exception(exc_type, exc_value, exc_traceback):
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).error("Unhandled exception")
    sys.excepthook = log_exception

    return log_filename


def training_report(tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene : Scene, gaussians, renderFunc, renderArgs,
                    training_dataset, testing_dataset, lpipsFunc, loss_dict=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
        if loss_dict is not None:
            for key, value in loss_dict.items():
                tb_writer.add_scalar(f'train_loss_patches/{key}', value.item(), iteration)

    psnr_test_iter = 0.0
    psnr_train_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        validation_configs = (
            {
                'name'    : 'train',
                'cameras' : [training_dataset[idx % len(training_dataset)] for idx in range(5, 30, 5)],
                'differentiable_cameras': scene.getTrainDifferentiableCameras(),
            },
            {
                'name'    : 'test',
                'cameras' : [testing_dataset[idx % len(testing_dataset)] for idx in range(5, 30, 5)],
                'differentiable_cameras': scene.getTestDifferentiableCameras(),
            }
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0

                differentiable_cams = config['differentiable_cameras']
                for idx, batch_data in enumerate(tqdm(config['cameras'])):
                    gt_image, viewpoint, _ = batch_data
                    gt_image = gt_image.cuda()
                    viewpoint = viewpoint.cuda()
                    differentiable_cam = differentiable_cams.get_camera(viewpoint.camera_id)
                    render_pkg = renderFunc(viewpoint, differentiable_cam, gaussians, *renderArgs, apply_distortion=False)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)

                    if idx < 5:
                        grid = [gt_image, image]
                        grid = make_grid(grid, nrow=1)
                        log_dir = os.path.join(scene.model_path, f"log/{config['name']}")
                        os.makedirs(log_dir, exist_ok=True)
                        image_name = viewpoint.image_name.split('.')[0]
                        if 'frame' in image_name:
                            timestamp = viewpoint.file_path.split('/')[1].split('_')[-1] # "preprocess/time_0030/diffusion_rcm3/frame_00036.png"
                            for i_distort, distortion_meta in enumerate(gaussians._distortion_meta):
                                if viewpoint.timestamp >= distortion_meta["start"] and viewpoint.timestamp < distortion_meta["end"]:
                                    break
                            try:
                                image_name = image_name + '_' + timestamp + '_' + str(i_distort)
                            except:
                                image_name = image_name + '_' + timestamp
                        torchvision.utils.save_image(grid[None], os.path.join(log_dir, f"{iteration}_"+image_name+f".png"))

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])

                logger.info(f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                if config['name'] == 'test':
                    psnr_test_iter = psnr_test.item()
                elif config['name'] == 'train':
                    psnr_train_iter = psnr_test.item()

    torch.cuda.empty_cache()
    return psnr_test_iter, psnr_train_iter

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--config", type=str,
                        help="Path to YAML config file. Values override argparse defaults.")
    parser.add_argument('--debug_from', type=int, default=-1,
                        help="Enable pipeline debug mode starting from this iteration. -1 to disable.")
    parser.add_argument('--detect_anomaly', action='store_true', default=False,
                        help="Enable torch autograd anomaly detection (slow, use for debugging NaNs).")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10, 1_000, 3_000, 7_000, 10_000, 15_000, 30_000],
                        help="Iterations at which to run evaluation and save visualizations.")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000],
                        help="Iterations at which to save a point cloud checkpoint. Final iteration is always appended.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-iteration stdout output.")
    parser.add_argument("--start_checkpoint", type=str, default=None,
                        help="Resume from a .pth checkpoint.")
    parser.add_argument("--resume_training", type=bool, default=True,
                        help="When loading a checkpoint, restore optimizer state and continue from saved iteration.")

    # These parameters are typically set via config; defaults here are only
    # used when running without a config file (e.g. quick CLI experiments).
    # action='store_true' flags cannot be None, so they default to False.
    parser.add_argument("--gaussian_dim", type=int, default=None,
                        help="Gaussian dimension: 3 for static 3DGS, 4 for dynamic 4DGS (adds temporal coordinate t).")
    parser.add_argument("--time_duration", nargs=2, type=float, default=None,
                        help="Time range [t_start, t_end] of the sequence before frame_ratio scaling.")
    parser.add_argument('--num_pts', type=int, default=None,
                        help="Number of initial Gaussian points sampled from the point cloud.")
    parser.add_argument('--num_pts_ratio', type=float, default=None,
                        help="Fraction of num_pts to actually use (useful for ablations).")
    parser.add_argument("--rot_4d", action="store_true",
                        help="Use 4D rotations (6DOF) for dynamic Gaussians. Only effective when gaussian_dim=4.")
    parser.add_argument("--force_sh_3d", action="store_true",
                        help="Force 3D spherical harmonics even in 4D mode (disables temporal SH).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Number of views rendered per gradient step. Gradients are averaged across the batch.")
    parser.add_argument("--seed", type=int, default=6666,
                        help="Random seed for reproducibility.")
    parser.add_argument("--target_cam", nargs="+", type=str, default=[],
                        help="Restrict training/testing to these camera names. Empty means all cameras.")
    parser.add_argument("--target_ts_train", nargs="+", type=int, default=None,
                        help="Target timestamp indices for training views. -1 loads all timestamps. "
                             "Two values [start_frame_id, end_frame_id] define a range; multiple values select specific frames.")
    parser.add_argument("--target_ts_test", nargs="+", type=int, default=None,
                        help="Target timestamp indices for test views. Same format as target_ts_train.")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    #! Merge Config
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if key == "kplanes_config" or key == "start_checkpoint":
            setattr(args, key, host[key])
        elif isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)

    # Apply fallback defaults for config-driven params if neither config nor
    # CLI provided a value (e.g. running without a full config file).
    _DEFAULTS = {
        "gaussian_dim":    4,
        "time_duration":   [0.0, 1.0],
        "num_pts":         300_000,
        "num_pts_ratio":   1.0,
        "batch_size":      1,
        "target_ts_train": [-1],
        "target_ts_test":  [0],
    }
    for _k, _v in _DEFAULTS.items():
        if getattr(args, _k) is None:
            setattr(args, _k, _v)

    setup_seed(args.seed)

    logger.info("Optimizing " + args.model_path)
    log_filename = prepare_logger(args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    TENSORBOARD_FOUND = False

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    interrupted_by_keyboard = False
    try:
        logger.info("Training started.")
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.resume_training, args.debug_from,
                args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size,
                args.target_cam, args.target_ts_train, args.target_ts_test, hyper=hp.extract(args))
    except KeyboardInterrupt:
        interrupted_by_keyboard = True
        logger.warning("Program interrupted by user (Ctrl+C).")

    finally:
        if interrupted_by_keyboard:
            if os.path.exists(log_filename):
                logger.info(f"Deleting log file {log_filename} due to user interrupt.")
                os.remove(log_filename)
        else:
            logger.info("Training complete; log file retained.")
