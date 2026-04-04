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
import torch
import random
import json
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.mix_gaussian_model import MixGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.data_utils import SharedCameraDataset
from scene.differentiable_cameras import DifferentiableCameras
from loguru import logger
from multiprocessing import Manager, Lock
import numpy as np

class Scene:
    def __init__(self, args : ModelParams, shuffle=False, resolution_scales=[1.0], num_pts=100_000, num_pts_ratio=1.0, time_duration=None,
                 target_cam=[], target_time_train=[-1], target_time_test=[0], cached_dataset=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.white_background = args.white_background
        self.args = args
        self.time_duration = time_duration
        self.target_cam = target_cam
        self.target_time_train = target_time_train
        self.target_time_test = target_time_test
        self.shuffle = shuffle
        logger.info(f"[TARGET] training target time: {self.target_time_train}, testing target time: {self.target_time_test} | target cam: {self.target_cam}")

        #! Load Scene Info from .json file
        if os.path.exists(os.path.join(args.source_path, args.transforms_file)):
            logger.info(f"Given transforms file {args.transforms_file}, assuming Blender(Ours) data set!")
            scene_info = sceneLoadTypeCallbacks["Blender_new"](args.source_path, args.white_background, args.eval,
                                                        num_pts=num_pts, time_duration=time_duration, extension=args.extension,
                                                        num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader,
                                                        target_cam=target_cam, transforms_file=args.transforms_file, ply_file=args.ply_file,
                                                        scene_scale_ratio=args.scene_scale_ratio, camera_downsample=args.camera_downsample)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            logger.info("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval,
                                                        num_pts=num_pts, time_duration=time_duration, extension=args.extension,
                                                        num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader,
                                                        target_cam=target_cam)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, num_pts_ratio=num_pts_ratio)
        else:
            assert False, "Could not recognize scene type!"
        self.scene_info = scene_info

        #! Backup point cloud file for initialization and camera info for training
        storePly(os.path.join(self.model_path, "input.ply"), scene_info.point_cloud.points, np.uint8(np.clip((scene_info.point_cloud.colors), 0, 1) * 255))
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        #! Get Camera Extent
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        logger.info(f"cameras extent = {self.cameras_extent}")

        #! Setup Cameras for training
        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)
        self.train_cameras = {}
        self.train_differentiable_cameras = {}
        self.test_cameras = {}
        self.test_differentiable_cameras = {}
        for resolution_scale in resolution_scales:
            logger.info("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            self.train_differentiable_cameras[resolution_scale] = DifferentiableCameras(scene_info.train_cameras, resolution_scale, args)
            logger.info("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            self.test_differentiable_cameras[resolution_scale] = DifferentiableCameras(scene_info.test_cameras, resolution_scale, args)
        self.n_differentiable_cameras = len(self.train_differentiable_cameras[1.0].unique_camera_ids)
        logger.info(f"{self.n_differentiable_cameras} training differentiable cameras loaded")
        logger.info(f"{len(self.test_differentiable_cameras[1.0].unique_camera_ids)} testing differentiable cameras loaded")

        #! Generate Dataset
        # # 创建共享缓存（在主进程中完成）
        manager = Manager()
        self.shared_cache = manager.dict()
        self.cache_lock = Lock()
        self.cached_dataset = cached_dataset
        self.train_camera_dataset = SharedCameraDataset(
            self.train_cameras[1.0].copy(), self.white_background,
            shared_cache=self.shared_cache, cache_lock=self.cache_lock,
            target_time=self.target_time_train, cache_activate=self.cached_dataset
        )
        self.test_camera_dataset = SharedCameraDataset(
            self.test_cameras[1.0].copy(), self.white_background,
            shared_cache=self.shared_cache, cache_lock=self.cache_lock,
            target_time=self.target_time_test, cache_activate=self.cached_dataset
        )

        logger.info(f"SharedCameraDataset created. Training Dataset: {len(self.train_camera_dataset)} views; Testing Dataset: {len(self.test_camera_dataset)} views")

    def save(self, gaussians:MixGaussianModel, iteration):
        torch.save((gaussians.capture(), iteration), self.model_path + "/chkpnt" + str(iteration) + ".pth")

    def getTrainCameras(self, scale=1.0, target_time=None):
        if target_time is None or target_time == self.target_time_train:
            return self.train_camera_dataset

        return SharedCameraDataset(
            self.train_cameras[scale].copy(), self.white_background,
            shared_cache=self.shared_cache, cache_lock=self.cache_lock,
            target_time=target_time, cache_activate=self.cached_dataset
        )

    def getTestCameras(self, scale=1.0, target_time=None):
        if target_time is None or target_time == self.target_time_test:
            return self.test_camera_dataset

        return SharedCameraDataset(
            self.test_cameras[scale].copy(), self.white_background,
            shared_cache=self.shared_cache, cache_lock=self.cache_lock,
            target_time=target_time, cache_activate=self.cached_dataset
        )

    def getTrainDifferentiableCameras(self, scale=1.0):
        return self.train_differentiable_cameras[scale]

    def getTestDifferentiableCameras(self, scale=1.0):
        return self.test_differentiable_cameras[scale]