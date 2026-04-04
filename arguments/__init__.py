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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.extension = ".png"
        self.num_extra_pts = 0
        self.frame_ratio = 1
        self.Ntime = 300
        self.dataloader = False

        self.lpips_spatial = True
        self.lpips_net = "alex"
        self.transforms_file = "cameras.json"
        self.ply_file = "mast3r_dense_pts_merged.ply"
        self.lpips_decay = False
        self.weighted_sample = True
        self.experiment_name = ""

        self.scene_scale_ratio = 1.0
        self.camera_downsample = 1 # Downsample Camera Poses

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.env_map_res = 0
        self.env_optimize_until = 1000000000
        self.env_optimize_from = 0
        self.eval_shfs_4d = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_t_lr_init = -1.0
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.thresh_opa_prune = 0.005
        self.densification_interval = 100
        self.opacity_reset_interval = 30000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.densify_grad_t_threshold = 0.0002 / 40
        self.densify_until_num_points = -1
        self.final_prune_from_iter = -1
        self.sh_increase_interval = 1000
        self.lambda_opa_mask = 0.0
        self.lambda_rigid = 0.0
        self.lambda_motion = 0.0

        self.lambda_lpips = 0.0
        self.lambda_depth = 0.0
        self.depth_l1_weight_init = 0.5
        self.depth_l1_weight_final = 0.001
        self.lambda_flow = 0.0
        self.flow_l1_weight_init = 0.1
        self.flow_l1_weight_final = 0.001
        self.lambda_hexplane = 1.0
        # Pearson Depth Loss
        self.depth_pearson_weight_init = 0.15
        self.depth_pearson_weight_final = 0.001
        self.box_p = 128
        self.p_corr = 0.5

        # #-------------------- Pose Optimization -------------------# #
        self.cam_optim_from_iter = 0
        self.cam_optim_until_iter = 7_000
        self.camera_lr_max_steps = 7_000
        self.rotation_lr_init   = 1e-4
        self.rotation_lr_final  = 1e-6
        self.translation_lr_init = 1e-3
        self.translation_lr_final = 1e-5
        self.lambda_pose = 0.0
        # #----------------------------------------------------------# #

        # #-------------------- Lifespane Regularization ------------# #
        self.lambda_lifespan = 0.0
        # #----------------------------------------------------------# #

        # #--------------------- Distortion Field -------------------# #
        self.deformation_lr_init = 0.00016 # 1.6e-4
        self.deformation_lr_final = 0.000016 # 1.6e-5
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.0016 # 1.6e-3
        self.grid_lr_final = 0.00016 # 1.6e-4
        # #----------------------------------------------------------# #
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64 # width of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.timebase_pe = 4
        self.defor_depth = 1 # depth of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.6
        self.zero_canonical_weight = 1.0
        self.plane_tv_weight = 0.0001 # TV loss of spatial grid
        self.time_smoothness_weight = 0.01 # TV loss of temporal grid
        self.l1_time_planes = 0.0001  # TV loss of temporal grid
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 150]  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
                            }
        self.multires = [1, 2, 4, 8] # multi resolution of voxel grid
        self.no_dx=False # cancel the deformation of Gaussians' position
        self.no_grid=False # cancel the spatial-temporal hexplane.
        self.no_ds=False # cancel the deformation of Gaussians' scaling
        self.no_dr=True # cancel the deformation of Gaussians' rotations
        self.no_do=True # cancel the deformation of Gaussians' opacity
        self.no_dshs=True # cancel the deformation of SH colors.
        self.empty_voxel=False
        self.grid_pe=0
        self.static_mlp=False
        self.apply_rotation=False

        self.train_distortion = False
        self.distortion_optim_from_iter = 5_000
        self.distortion_optim_until_iter = -1

        #* FiLM modulate
        self.apply_film_modulate = False
        super().__init__(parser, "ModelHiddenParams")