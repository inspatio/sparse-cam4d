import torch
from torch.utils.data import Dataset, DataLoader
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np
from loguru import logger

class SharedCameraDataset(Dataset):
    def __init__(self, viewpoint_stack, white_background, shared_cache, cache_lock, target_time=[-1], cache_activate=True):
        self.cache_activate = cache_activate
        if cache_activate:
            self.shared_cached_image = shared_cache
            self.cache_lock = cache_lock  # 多线程安全保护
        self.target_time = target_time
        self.viewpoint_stack_all = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        self.set_target_viewpoint_stack(self.target_time)


    def set_target_viewpoint_stack(self, target_time):
        self.target_time = target_time
        try:
            if len(target_time) == 1 and target_time[0] < 0:
                logger.info(f"Load ALL Viewpoints.")
                self.viewpoint_stack = self.viewpoint_stack_all
            elif len(target_time) == 2:
                start_ts = target_time[0]
                end_ts = target_time[1]
                max_timestamp = max([v.timestamp for v in self.viewpoint_stack_all])
                logger.info(f"max timestamp = {max_timestamp}, target time duration: {target_time}")
                self.viewpoint_stack = [viewpoint for viewpoint in self.viewpoint_stack_all
                                        if (viewpoint.timestamp > start_ts or np.isclose(viewpoint.timestamp, start_ts)) and
                                            (viewpoint.timestamp < end_ts or np.isclose(viewpoint.timestamp, end_ts))
                                        ]
            else:
                max_timestamp = max([v.timestamp for v in self.viewpoint_stack_all])
                logger.info(f"max timestamp = {max_timestamp}, target time: {target_time}")
                self.viewpoint_stack = [viewpoint for viewpoint in self.viewpoint_stack_all if any(np.isclose(viewpoint.timestamp, tt) for tt in target_time)]
        except:
            logger.warning(f"Invalid target time {target_time}, load all viewpoint")
            self.viewpoint_stack = self.viewpoint_stack_all

        self.true_viewpoint_idx = [idx for idx, v in enumerate(self.viewpoint_stack) if v.is_true_image]

    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if viewpoint_cam.meta_only:
            if self.cache_activate:
                # Read data: atomic operation
                cached = self.shared_cached_image.get(viewpoint_cam.image_path, None)
                if cached is not None:
                    return cached[0], viewpoint_cam, cached[1]

            viewpoint_image, gt_alpha_mask = self.load_image(viewpoint_cam)

            # 配合sampler，只有真实图像才缓存
            if self.cache_activate and viewpoint_cam.is_true_image:
                with self.cache_lock:
                    # double check to avoid re-write
                    if viewpoint_cam.image_path not in self.shared_cached_image:
                        self.shared_cached_image[viewpoint_cam.image_path] = (viewpoint_image, gt_alpha_mask)
            return viewpoint_image, viewpoint_cam, gt_alpha_mask
        else:
            viewpoint_image = viewpoint_cam.image
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask
            return viewpoint_image, viewpoint_cam, gt_alpha_mask

    def __len__(self):
        return len(self.viewpoint_stack)

    def load_image(self, viewpoint_cam):
        with Image.open(viewpoint_cam.image_path) as image_load:
            im_data = np.array(image_load.convert("RGBA"))
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
        #! process alpha channel
        if norm_data[:, :, 3:4].min() < 1:
            arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
            image_load = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
            resized_image = PILtoTorch(image_load, viewpoint_cam.resolution)
            viewpoint_image = resized_image[:3, ...].clamp(0.0, 1.0)
            gt_alpha_mask = resized_image[3:4, ...]
        else:
            image_load = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            resized_image = PILtoTorch(image_load, viewpoint_cam.resolution)
            viewpoint_image = resized_image[:3, ...].clamp(0.0, 1.0)
            gt_alpha_mask = torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))

        return viewpoint_image, gt_alpha_mask

def generate_dataloader(dataset, scene, gaussians, batch_size, train_target_time, test_target_time):
    training_dataset = scene.getTrainCameras(target_time=train_target_time)
    testing_dataset = scene.getTestCameras(target_time=test_target_time)
    logger.info(f"Create Dataset with {len(training_dataset)} training data, {len(testing_dataset)} testing data")
    logger.info(f"==> Training target time = {training_dataset.target_time}")
    logger.info(f"==> Testing target time = {testing_dataset.target_time}")

    if dataset.weighted_sample:
        #* Real:Gen = 1:1 Random Sample Data Loading
        logger.info("Real:Gen = 1:1 Random Sample Data Loading")
        viewpoint_stack = training_dataset.viewpoint_stack
        sampler = EpochAwareStratifiedSampler(
            data_source = training_dataset,
            pseudo_mask = torch.tensor([not cam.is_true_image for cam in viewpoint_stack], dtype=torch.bool),
            timestamps  = torch.tensor([cam.timestamp for cam in viewpoint_stack], dtype=torch.float),
            view_ids = torch.tensor([cam.camera_id for cam in viewpoint_stack], dtype=torch.int),
            time_bin_size = 15*gaussians.min_timestep,
            repeat_real = True,
        )
        logger.info(f"Epoch Aware Stratified Sampler inited")

    else:
        sampler = None
        logger.info(f"No Sampler inited")
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, collate_fn=lambda x: x, drop_last=True, sampler=sampler,
                                     num_workers=16 if dataset.dataloader else 0, pin_memory=True, prefetch_factor=16 if dataset.dataloader else None)
    logger.info(f"DataLoader create, len = {len(training_dataloader)}, batch size = {batch_size}")

    return training_dataset, testing_dataset, training_dataloader, sampler

import random
from collections import defaultdict
class EpochAwareStratifiedSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, pseudo_mask, timestamps, view_ids,
                 time_bin_size=15, repeat_real=True, seed=0):
        self.data_source = data_source
        self.pseudo_mask = torch.as_tensor(pseudo_mask, dtype=torch.bool)
        self.real_mask = ~self.pseudo_mask
        self.timestamps = torch.as_tensor(timestamps)
        self.view_ids = torch.as_tensor(view_ids)
        self.time_bin_size = time_bin_size
        self.pseudo_per_epoch = len(data_source) // 2
        self.repeat_real = repeat_real
        self.seed = seed

        self.real_idxs = torch.where(self.real_mask)[0].tolist()
        self.pseudo_idxs = torch.where(self.pseudo_mask)[0].tolist()

        self.pseudo_bins = defaultdict(list)
        for idx in self.pseudo_idxs:
            t_bin = int(self.timestamps[idx] // self.time_bin_size)
            self.pseudo_bins[t_bin].append(idx)

        self.real_bins = defaultdict(list)
        for idx in self.real_idxs:
            t_bin = int(self.timestamps[idx] // self.time_bin_size)
            self.real_bins[t_bin].append(idx)

        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.pseudo_per_epoch * 2 if self.repeat_real else self.pseudo_per_epoch + len(self.real_idxs)

    def _stratified_sample(self, bins, total, rng):
        bin_keys = sorted(bins.keys())
        per_bin = max(1, total // len(bin_keys))
        sampled = []
        for b in bin_keys:
            bin_indices = bins[b]
            if len(bin_indices) <= per_bin:
                sampled.extend(bin_indices)
            else:
                chosen = rng.sample(bin_indices, k=per_bin)
                sampled.extend(chosen)
        if len(sampled) < total:
            sampled.extend(rng.choices(sampled, k=total - len(sampled)))
        return sampled[:total]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        pseudo_sampled = self._stratified_sample(self.pseudo_bins, self.pseudo_per_epoch, rng)

        if self.repeat_real:
            real_sampled = self._stratified_sample(self.real_bins, self.pseudo_per_epoch, rng)
        else:
            real_sampled = self.real_idxs

        full_batch = pseudo_sampled + real_sampled
        rng.shuffle(full_batch)
        return iter(full_batch)