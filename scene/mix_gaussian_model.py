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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_rotation_4d, build_scaling_rotation_4d
from torch import nn
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import sh_channels_4d
from scene.deformation import deform_network
from loguru import logger
import torch.nn.functional as F

class MixGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L.transpose(1, 2) @ L
            symm = strip_symmetric(actual_covariance)
            return symm

        def build_covariance_from_scaling_rotation_4d(scaling, scaling_modifier, rotation_l, rotation_r, dt=0.0):
            L = build_scaling_rotation_4d(scaling_modifier * scaling, rotation_l, rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            cov_11 = actual_covariance[:,:3,:3]
            cov_12 = actual_covariance[:,0:3,3:4]
            cov_t = actual_covariance[:,3:4,3:4]
            current_covariance = cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t
            symm = strip_symmetric(current_covariance)
            if dt.shape[1] > 1:
                mean_offset = (cov_12.squeeze(-1) / cov_t.squeeze(-1))[:, None, :] * dt[..., None]
                mean_offset = mean_offset[..., None]  # [num_pts, num_time, 3, 1]
            else:
                mean_offset = cov_12.squeeze(-1) / cov_t.squeeze(-1) * dt
            return symm, mean_offset.squeeze(-1)

        self.scaling_activation = F.softplus
        self.scaling_inverse_activation = lambda y: torch.log(torch.expm1(y) + 1e-6)

        if not self.rot_4d:
            self.covariance_activation = build_covariance_from_scaling_rotation
        else:
            self.covariance_activation = build_covariance_from_scaling_rotation_4d

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def setup_distortion_fields(self, target_ts_train, piece_length):
        '''
        Setup distortion fields for training. Supports training with multiple distortion fields, each with `piece_length` frames.
        '''
        i = 0
        while True:
            train_target_time = [
                    (target_ts_train[0] + piece_length * i) * self.min_timestep,
                    (min(target_ts_train[0] + piece_length * (i + 1), target_ts_train[1])) * self.min_timestep
                ]
            if not (train_target_time[0] < train_target_time[1]):
                break
            distortion_meta = {'start': train_target_time[0], 'end': train_target_time[1]}
            self._distortion_meta.append(distortion_meta)
            self._distortion_list.append(deform_network(self.distortion_args).cuda())
            i += 1


    def __init__(self, sh_degree : int, gaussian_dim : int = 3, time_duration: list = [-0.5, 0.5], rot_4d: bool = False, force_sh_3d: bool = False, sh_degree_t : int = 0,
                 distortion_args = None, frame_ratio=1.0, Ntime=300.0):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.gaussian_dim = gaussian_dim
        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self._rotation_r = torch.empty(0)
        self.force_sh_3d = force_sh_3d
        self.t_gradient_accum = torch.empty(0)
        if self.rot_4d or self.force_sh_3d:
            assert self.gaussian_dim == 4
        self.env_map = torch.empty(0)

        self.active_sh_degree_t = 0
        self.max_sh_degree_t = sh_degree_t

        # #---------------------------- Distortion Field --------------------# #
        self.distortion_args = distortion_args
        self._distortion_list = nn.ModuleList()
        self._distortion_meta = []
        # #------------------------------------------------------------------# #
        self._grad_hooks = {}  # 保存所有张量的 hook handle，方便梯度管理

        self.frame_ratio = frame_ratio
        self.min_timestep = 1.0/Ntime/frame_ratio
        self.setup_functions()

    def capture(self):
        if self.gaussian_dim == 3:
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
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self._distortion_meta,
                self._distortion_list.state_dict(),
            )
        elif self.gaussian_dim == 4:
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
                self.t_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self._t,
                self._scaling_t,
                self._rotation_r,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t,
                self._distortion_meta,
                self._distortion_list.state_dict(),
            )

    def restore(self, model_args, training_args, differentiable_cameras=None, iteration=-1, resume_training=True):
        if self.gaussian_dim == 3:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self._distortion_meta,
            distort_dict) = model_args
        elif self.gaussian_dim == 4:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            t_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self._t,
            self._scaling_t,
            self._rotation_r,
            self.rot_4d,
            self.env_map,
            self.active_sh_degree_t,
            self._distortion_meta,
            distort_dict) = model_args

        N_distortion = len(self._distortion_meta)
        if N_distortion == 0:
            logger.info("Training without Distortion Field!")
        else:
            for _ in range(N_distortion):
                self._distortion_list.append(deform_network(self.distortion_args).cuda())

        if training_args is not None:
            self.training_setup(training_args, differentiable_cameras)

            if resume_training:
                self.xyz_gradient_accum = xyz_gradient_accum
                self.t_gradient_accum = t_gradient_accum if self.gaussian_dim==4 else None
                self.denom = denom
                self.optimizer.load_state_dict(opt_dict)
        # #-------------------------- Distortion Field ---------------------------# #
        self._distortion_list.load_state_dict(distort_dict)
        logger.info(f"Restoring Distortion Field from Iteration {iteration} with {len(self._distortion_list)} distortion fields")

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        l.append('t')
        for i in range(self._scaling_t.shape[1]):
            l.append('scale_t_{}'.format(i))
        for i in range(self._rotation_r.shape[1]):
            l.append('rot_r_{}'.format(i))

        return l

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)

    @property
    def get_scaling_xyzt(self):
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim = 1))

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_rotation_r(self):
        return self.rotation_activation(self._rotation_r)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_t(self):
        return self._t

    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim = 1)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.max_sh_degree+1)**2
        elif self.gaussian_dim == 4 and self.max_sh_degree_t == 0:
            return sh_channels_4d[self.max_sh_degree]
        elif self.gaussian_dim == 4 and self.max_sh_degree_t > 0:
            return (self.max_sh_degree+1)**2 * (self.max_sh_degree_t + 1)

    def get_velocity(self, scaling_modifier = 1):
        L = build_scaling_rotation_4d(scaling_modifier * self.get_scaling_xyzt, self._rotation, self._rotation_r)
        actual_covariance = L @ L.transpose(1, 2)
        cov_12 = actual_covariance[:,0:3,3:4]
        cov_t = actual_covariance[:,3:4,3:4]
        velocity = cov_12.squeeze(-1) / cov_t.squeeze(-1)
        return velocity

    def get_cov_t(self, scaling_modifier = 1):
        if self.rot_4d:
            L = build_scaling_rotation_4d(scaling_modifier * self.get_scaling_xyzt, self._rotation, self._rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance[:,3,3].unsqueeze(1)
        else:
            return self.get_scaling_t * scaling_modifier

    def get_marginal_t(self, timestamp, scaling_modifier = 1): # Standard
        sigma = self.get_cov_t(scaling_modifier)
        return torch.exp(-0.5*(self.get_t-timestamp)**2/sigma) # / torch.sqrt(2*torch.pi*sigma)

    def get_lifespan(self, scaling_modifier = 1, opacity_th=torch.tensor(0.05)):
        sigma = self.get_cov_t(scaling_modifier)
        return torch.sqrt( torch.log(opacity_th) / -0.5 * sigma ) * 2

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_current_covariance_and_mean_offset(self, scaling_modifier = 1, timestamp = 0.0):
        return self.covariance_activation(self.get_scaling_xyzt, scaling_modifier,
                                                              self._rotation,
                                                              self._rotation_r,
                                                              dt = timestamp - self.get_t)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        elif self.max_sh_degree_t and self.active_sh_degree_t < self.max_sh_degree_t:
            self.active_sh_degree_t += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(pcd.time).cuda().float()
        logger.info(f"Number of points at initialization: {fused_point_cloud.shape[0]}")

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            dist_t = torch.zeros_like(fused_times, device="cuda") + (self.time_duration[1] - self.time_duration[0]) / 5
            scales_t = torch.log(torch.sqrt(dist_t))
            if self.rot_4d:
                rots_r = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots_r[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def training_setup(self, training_args, differentiable_cameras=None):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        #! Optimizable Parameters
        #* 4D Gaussian Parameters
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.gaussian_dim == 4:
            if training_args.position_t_lr_init < 0:
                training_args.position_t_lr_init = training_args.position_lr_init
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            l.append({'params': [self._t], 'lr': training_args.position_t_lr_init * self.spatial_lr_scale, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': training_args.scaling_lr * 10.0, "name": "scaling_t"})
            if self.rot_4d:
                l.append({'params': [self._rotation_r], 'lr': training_args.rotation_lr, "name": "rotation_r"})
        #* Pose Optimization
        if differentiable_cameras is not None:
            l_cam = [
                {'params': [differentiable_cameras.get_camera(cid).q for cid in differentiable_cameras.cameras], 'lr': training_args.rotation_lr_init, 'name': f'cam_rotation'},
                {'params': [differentiable_cameras.get_camera(cid).T for cid in differentiable_cameras.cameras], 'lr': training_args.translation_lr_init, 'name': f'cam_translation'},
            ]
            l+=l_cam
        #* Distortion Fields
        if hasattr(self, "_distortion_list") and self._distortion_list is not None:
            for i, distortion in enumerate(self._distortion_list):
                l += [
                    {
                        'params': list(distortion.get_mlp_parameters()),
                        'lr': training_args.deformation_lr_init * self.spatial_lr_scale,
                        'name': f'deformation_{i}'
                    },
                    {
                        'params': list(distortion.get_grid_parameters()),
                        'lr': training_args.grid_lr_init * self.spatial_lr_scale,
                        'name': f'grid_{i}'
                    }
                ]
        #* Optimizer
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        #! Learning Rate Scheduler
        #* 4D Gaussian Parameters
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        if self.gaussian_dim == 4:
            self.scale_t_scheduler_args = get_expon_lr_func(
                                                    lr_init=training_args.scaling_lr*10.0,
                                                    lr_final=training_args.scaling_lr,
                                                    max_steps=7000)
        else:
            self.scale_t_scheduler_args = None
        #* Pose Optimization
        if differentiable_cameras is not None:
            self.cam_rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr_init,
                                                                lr_final=training_args.rotation_lr_final,
                                                                max_steps=training_args.camera_lr_max_steps)
            self.cam_translation_scheduler_args = get_expon_lr_func(lr_init=training_args.translation_lr_init,
                                                                    lr_final=training_args.translation_lr_final,
                                                                    max_steps=training_args.camera_lr_max_steps)
        #* Distortion Fields
        if hasattr(self, "_distortion_list") and self._distortion_list is not None:
            self.deformation_scheduler_args_list = []
            self.grid_scheduler_args_list = []
            for _ in self._distortion_list:
                deformation_args = get_expon_lr_func(
                    lr_init=training_args.deformation_lr_init * self.spatial_lr_scale,
                    lr_final=training_args.deformation_lr_final * self.spatial_lr_scale,
                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                    max_steps=training_args.position_lr_max_steps
                )
                grid_args = get_expon_lr_func(
                    lr_init=training_args.grid_lr_init * self.spatial_lr_scale,
                    lr_final=training_args.grid_lr_final * self.spatial_lr_scale,
                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                    max_steps=training_args.position_lr_max_steps
                )
                self.deformation_scheduler_args_list.append(deformation_args)
                self.grid_scheduler_args_list.append(grid_args)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        update_lr = {}
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                update_lr["xyz"] = lr
            if param_group["name"] == "scaling_t" and self.scale_t_scheduler_args is not None:
                lr = self.scale_t_scheduler_args(iteration)
                param_group['lr'] = lr
                update_lr["scaling_t"] = lr
            if param_group["name"] == "cam_rotation":
                lr = self.cam_rotation_scheduler_args(iteration)
                param_group['lr'] = lr
                update_lr["cam_rotation"] = lr
            if param_group["name"] == "cam_translation":
                lr = self.cam_translation_scheduler_args(iteration)
                param_group['lr'] = lr
                update_lr["cam_translation"] = lr

            if "deformation" in param_group["name"]:
                name = param_group["name"]
                idx = int(name.split("deformation_")[1]) if "deformation_" in name else 0
                lr = self.deformation_scheduler_args_list[idx](max(iteration-self.distortion_args.distortion_optim_from_iter, 0))
                param_group['lr'] = lr
                update_lr[f"deformation_{idx}"] = lr

            if "grid" in param_group["name"]:
                name = param_group["name"]
                idx = int(name.split("grid_")[1]) if "grid_" in name else 0
                lr = self.grid_scheduler_args_list[idx](max(iteration-self.distortion_args.distortion_optim_from_iter, 0))
                param_group['lr'] = lr
                update_lr[f"grid_{idx}"] = lr
        return update_lr

    def reset_opacity(self, selected_mask=None):
        if selected_mask is not None:
            opacities_new = self.get_opacity
            opacities_new[selected_mask] = torch.min(opacities_new[selected_mask], torch.ones_like(opacities_new[selected_mask])*0.01)
            opacities_new = inverse_sigmoid(opacities_new)
        else:
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in ["cam_rotation", "cam_translation"] or len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                try:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                except:
                    print(group['name'], stored_state["exp_avg"].shape, mask.shape)
                    raise ValueError
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

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

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not group["name"] in tensors_dict:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz, new_features_dc, new_features_rest, new_opacities,
        new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r
    ):
        d = {"xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
        }
        if self.gaussian_dim == 4:
            d["t"] = new_t
            d["scaling_t"] = new_scaling_t
            if self.rot_4d:
                d["rotation_r"] = new_rotation_r

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # logger.info(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        if not self.rot_4d:
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            if self.gaussian_dim == 4:
                stds_t = self.get_scaling_t[selected_pts_mask].repeat(N,1)
                means_t = torch.zeros((stds_t.size(0), 1),device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        else:
            stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 4),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation_4d(self._rotation[selected_pts_mask], self._rotation_r[selected_pts_mask]).repeat(N,1,1)
            new_rotation_r = self._rotation_r[selected_pts_mask].repeat(N,1)
            new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyzt[selected_pts_mask].repeat(N, 1)
            new_xyz = new_xyzt[...,0:3]
            new_t = new_xyzt[...,3:4]
            new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # logger.info(f"num_to_densify_pos: {torch.where(grads >= grad_threshold, True, False).sum()}, num_to_clone_pos: {selected_pts_mask.sum()}")

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None, prune_only=False):
        prune_mask_opacity = (self.get_opacity < min_opacity).squeeze()
        # logger.info(f"opacity min = {self.get_opacity.min().item()}, max = {self.get_opacity.max().item()}, mean = {self.get_opacity.mean().item()}, median = {self.get_opacity.median().item()}")
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask_opacity, big_points_vs), big_points_ws)
        else:
            prune_mask = prune_mask_opacity
        self.prune_points(prune_mask)
        # logger.info(f"num_to_prune: {prune_mask.sum()}, opacity prune: {prune_mask_opacity.sum()}, scale prune: {prune_mask.sum()-prune_mask_opacity.sum()} | max scale: {self.get_scaling.max().item()}")

        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            if self.gaussian_dim == 4:
                grads_t = self.t_gradient_accum / self.denom
                grads_t[grads_t.isnan()] = 0.0
            else:
                grads_t = None
            self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t)
            self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t)
        else:
            if self.gaussian_dim == 4:
                self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]

    def add_densification_stats_grad(self, viewspace_point_grad, update_filter, avg_t_grad=None, denom=1):
        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.denom[update_filter] += denom # fix denomination bug in batch mode
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]

    # #-------------------------- Distortion Field ---------------------------# #
    def compute_plane_smoothness(self, t, mode='l2'):
        batch_size, c, h, w = t.shape
        # Convolve with a second derivative filter, in the time dimension which is dimension 2
        first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
        second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
        # assert torch.isnan(second_difference).any(), second_difference
        # Take the L2 norm of the result
        if mode == 'l2':
            return torch.square(second_difference).mean()
        elif mode == 'l1':
            return torch.abs(first_difference).mean()
        else:
            raise ValueError(f"Invalid smoothness mode {mode}")
    def _plane_regulation(self):
        total = 0
        for distortion in self._distortion_list:
            multi_res_grids = distortion.deformation_net.grid.grids
            # model.grids is 6 x [1, rank * F_dim, reso, reso]
            for grids in multi_res_grids:
                if len(grids) == 3:
                    time_grids = []
                else:
                    # time_grids =  [0,1,3] # 6-planes (xy xz xt yz yt zt)
                    time_grids =  [0,1,4] # 9-planes (xy xz xt xv yz yt yv zt zv)
                for grid_id in time_grids:
                    total += self.compute_plane_smoothness(grids[grid_id])
        return total
    def _time_regulation(self, mode='l2'):
        total = 0
        for distortion in self._distortion_list:
            multi_res_grids = distortion.deformation_net.grid.grids
            # model.grids is 6 x [1, rank * F_dim, reso, reso]
            for grids in multi_res_grids:
                if len(grids) == 3:
                    time_grids = []
                else:
                    # time_grids =[2, 4, 5] # 6-planes (xy xz xt yz yt zt)
                    time_grids =[3, 6, 8] # 9-planes (xy xz xt xv yz yt yv zt zv)
                for grid_id in time_grids:
                    total += self.compute_plane_smoothness(grids[grid_id], mode=mode)
        return total
    def _l1_regulation(self):
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        total = 0.0
        for distortion in self._distortion_list:
            multi_res_grids = distortion.deformation_net.grid.grids
            for grids in multi_res_grids:
                if len(grids) == 3:
                    continue
                else:
                    # These are the spatiotemporal grids
                    # spatiotemporal_grids = [2, 4, 5] # 6-planes (xy xz xt yz yt zt)
                    spatiotemporal_grids = [2, 5, 7, 3, 6, 8] # 9-planes (xy xz xt xv yz yt yv zt zv)
                for grid_id in spatiotemporal_grids:
                    total += torch.abs(1 - grids[grid_id]).mean()
        return total
    def compute_regulation_bkp(self, time_smoothness_weight, l1_time_planes_weight):
        regulation = 0
        if time_smoothness_weight > 0:
            time_reg = self._time_regulation(mode='l1')
            regulation += time_smoothness_weight * time_reg
            if torch.isnan(time_reg):
                logger.warning("time reg loss is nan")
        if l1_time_planes_weight > 0:
            l1_reg = self._l1_regulation()
            regulation += l1_time_planes_weight * l1_reg
            if torch.isnan(l1_reg):
                logger.warning("l1 time plane reg loss is nan")

        return regulation

    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        # plane_reg = self._plane_regulation()
        # time_reg = self._time_regulation()
        # l1_reg = self._l1_regulation()
        # regulation = plane_tv_weight * plane_reg + time_smoothness_weight * time_reg + l1_time_planes_weight * l1_reg

        regulation = torch.tensor(0.0, device='cuda')
        if time_smoothness_weight > 0:
            time_reg = self._time_regulation()
            time_smoothness = time_smoothness_weight * time_reg
            regulation += time_smoothness
            if torch.isnan(time_reg):
                logger.warning("time reg loss is nan")
        else:
            time_smoothness = torch.tensor(0.0, device='cuda')

        if l1_time_planes_weight > 0:
            l1_reg = self._l1_regulation()
            l1_time_planes = l1_time_planes_weight * l1_reg
            regulation += l1_time_planes
            if torch.isnan(l1_reg):
                logger.warning("l1 time plane reg loss is nan")
        else:
            l1_time_planes = torch.tensor(0.0, device='cuda')

        if plane_tv_weight > 0:
            plane_reg = self._plane_regulation()
            plane_tv = plane_tv_weight * plane_reg
            regulation += plane_tv
            if torch.isnan(plane_reg):
                logger.warning("plane reg loss is nan")
        else:
            plane_tv = torch.tensor(0.0, device='cuda')

        return regulation, (time_smoothness, l1_time_planes, plane_tv)
    # #-----------------------------------------------------------------------# #