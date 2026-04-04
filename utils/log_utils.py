from loguru import logger
import torch
import numpy as np
import collections
from matplotlib.colors import ListedColormap

class LPIPSSpikeDetector:
    def __init__(self, window_size=100, std_factor=3.0, min_count=10, device='cpu'):
        self.window_size = window_size
        self.std_factor = std_factor
        self.min_count = min_count
        self.device = device

        # 用 tensor 存储历史值，避免频繁 item()/np 转换
        self.values = collections.deque(maxlen=window_size)

    def update(self, value: torch.Tensor):
        """记录新的 LPIPS loss 值"""
        if value.numel() != 1:
            raise ValueError("Expected scalar tensor for LPIPS loss.")
        self.values.append(value.detach().to(self.device).float())

        values_tensor = torch.stack(list(self.values))  # [window_size]
        self.mean = values_tensor.mean()
        self.std = values_tensor.std()
        self.threshold = self.mean + self.std_factor * self.std

    def is_spike(self, value: torch.Tensor):
        """判断当前 loss 是否为 spike"""
        self.update(value)
        if len(self.values) < self.min_count:
            return False  # 不足以判断
        return value > self.threshold + 1e-3

    def get_stats(self):
        """返回 (threshold, mean, std) 三元组，方便调试和记录"""
        if len(self.values) < self.min_count:
            return None, None, None
        return self.threshold.item(), self.mean.item(), self.std.item()

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
def analyze_sampling_distribution(dataloader, num_cameras, time_duration, time_bin_size=15, save_path=None):
    """
    统计 dataloader 中每个 view × time bin 被采样的频率。

    Args:
        dataloader: PyTorch DataLoader
        num_cameras: 摄像机数量
        time_bin_size: 每个时间 bin 的跨度
        min_time: 时间下界（通常为0）
        max_time: 时间上界（如300帧）

    Returns:
        freq_matrix: numpy array of shape (num_views, num_bins)
    """
    min_time = time_duration[0]
    max_time = time_duration[1]
    time_bin_size = time_bin_size
    num_bins = int((max_time - min_time) / time_bin_size)
    freq_matrix = np.zeros((num_cameras, num_bins), dtype=int)

    for batch in tqdm(dataloader):
        for batch_data in batch:
            _, viewpoint_cam, _ = batch_data
            camera_id = viewpoint_cam.camera_id
            time = viewpoint_cam.timestamp
            if min_time <= time < max_time:
                time_bin = int((time - min_time) / time_bin_size)
                freq_matrix[camera_id, time_bin] += 1

    if save_path:
        # 可视化热力图
        plt.figure(figsize=(15, 6))
        # sns.heatmap(freq_matrix, annot=True, fmt="d", cmap="Blues")

        # 设置值为 0 的地方为红色，其余使用蓝色
        blues = sns.color_palette("Blues", n_colors=256)
        red_blues = ListedColormap(["red"] + blues[1:])
        sns.heatmap(freq_matrix, annot=False, fmt="d", cmap=red_blues, mask=False, cbar=True)

        plt.xlabel(f"Time Bin (every {time_bin_size} frames)")
        plt.ylabel("View ID")
        plt.title("Sampling Frequency: View ID vs Time Bin")
        plt.tight_layout()
        plt.show()
        plt.savefig(save_path)

    return freq_matrix

from time import time
class TimeLogger():
    def __init__(self):
        torch.cuda.synchronize()
        self.start_time = time()
        self.time_logger = {}
        self.enable = True

    def update(self, name):
        if self.enable:
            torch.cuda.synchronize()
            self.time_logger[name] = time() - self.start_time
            self.start_time = time()

    def show(self):
        if self.enable:
            torch.cuda.synchronize()
            logger.info("Efficiency Analysis:")
            total_time = sum(time for time in self.time_logger.values())
            for name, time in self.time_logger.items():
                logger.info(f"  {name}: {time} ({time/total_time*100:.2f}%)")
            logger.info(f"  Total time: {total_time}")

    def reset(self):
        if self.enable:
            torch.cuda.synchronize()
            self.time_logger = {}
            self.start_time = time()

    def enabled(self, enable):
        self.enable = enable