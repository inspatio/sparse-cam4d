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
from copy import deepcopy

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class Camera:
    def __init__(self, colmap_id, camera_id, image, gt_alpha_mask,
                 image_name, uid, timestamp,
                 data_device = "cuda",
                 depth=None, resolution=None, image_path=None,
                 meta_only=False, is_true_image=True, pseudo_timestamp=0.0,
):
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")
        # #----------------------- Image Related ----------------------# #
        self.uid = uid
        self.timestamp  = timestamp
        self.pseudo_timestamp = pseudo_timestamp
        self.file_path = image_name
        self.image_name = image_name.split('/')[-1]
        self.resolution = resolution
        self.image_path = image_path
        self.image = image
        self.gt_alpha_mask = gt_alpha_mask
        self.meta_only = meta_only
        self.is_true_image = is_true_image
        self.depth = depth
        self.image_width = resolution[0]
        self.image_height = resolution[1]

        # #-------------- Pose Optimization ------------------# #
        self.colmap_id = colmap_id # cam_info.uid, frame['id']
        self.camera_id = camera_id

    @property
    def distort_time(self):
        return torch.tensor([self.timestamp, self.pseudo_timestamp]).float()

    def cuda(self):
        cuda_copy = deepcopy(self)
        for k, v in cuda_copy.__dict__.items():
            if isinstance(v, torch.Tensor) and v.device != cuda_copy.data_device:
                cuda_copy.__dict__[k] = v.to(cuda_copy.data_device)
        return cuda_copy