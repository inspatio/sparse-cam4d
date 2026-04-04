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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image import StructuralSimilarityIndexMeasure

def l1_loss(network_output, gt, mask=None):
    if mask is None:
        return torch.abs((network_output - gt)).mean()
    else:
        mask = mask.expand_as(network_output).type_as(network_output)
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        # 扩展 mask 通道以匹配 ssim_map
        mask = mask.expand_as(ssim_map).type_as(ssim_map)
        ssim_map = ssim_map * mask

        if size_average:
            ssim_map_mean = ssim_map.sum() / (mask.sum() + 1e-8)
        else:
            B = ssim_map.shape[0]
            ssim_map_mean = ssim_map.view(B, -1).sum(dim=1) / (mask.view(B, -1).sum(dim=1) + 1e-8)
    else:
        if size_average:
            ssim_map_mean = ssim_map.mean()
        else:
            ssim_map_mean = ssim_map.mean(1).mean(1).mean(1)


    return ssim_map_mean

ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
def msssim(rgb, gts):
    # assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    # assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return ms_ssim(rgb, gts).item()

"""
A wrapper class for the perceptual deep feature loss.

Reference:
    Richard Zhang et al. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. (CVPR 2018).
"""
import lpips
import torch.nn as nn
class PerceptualLoss(nn.Module):
    def __init__(self, net='alex', device='cuda', spatial=False):
        super().__init__()
        self.model = lpips.LPIPS(net=net, verbose=False, spatial=spatial).to(device)
        self.device = device

    def get_device(self, default_device=None):
        """
        Returns which device module is on, assuming all parameters are on the same GPU.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return default_device

    def __call__(self, pred, target, normalize=True, mask=None):
        """
        Pred and target are Variables.
        If normalize is on, scales images between [-1, 1]
        Assumes the inputs are in range [0, 1].
        B 3 H W
        """
        if pred.shape[1] != 3:
            pred = pred.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)
        # print(pred.shape, target.shape)
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        # temp_device = pred.device
        # device = self.get_device(temp_device)

        device = self.device

        pred = pred.to(device).float()
        target = target.to(device)
        dist = self.model.forward(pred, target).squeeze(0)

        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=dist.shape[-2:], mode='bilinear', align_corners=False)[0]
            dist = dist * mask

        return dist.to(device)

import matplotlib.pyplot as plt
def visualize_loss(loss_map, gt_image):
    loss_map_norm = (loss_map - loss_map.min()) / (loss_map.max() - loss_map.min() + 1e-8)
    loss_map_np = loss_map_norm.detach().cpu().numpy()
    loss_colormap = plt.get_cmap("plasma")(loss_map_np)[:, :, :3]  # 去掉 alpha 通道
    loss_colormap = torch.from_numpy(loss_colormap).permute(2, 0, 1).float().to(gt_image.device)  # [3, H, W]
    if loss_colormap.shape[1:] != gt_image.shape[1:]:
        loss_colormap = nn.functional.interpolate(loss_colormap[None], size=gt_image.shape[1:], mode='bilinear', align_corners=False)[0]
    return loss_colormap

import math
def pearson_depth_loss(depth_src, depth_target):
    #co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))

    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co


def local_pearson_loss(depth_src, depth_target, box_p, p_corr):
    # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
    num_box_h = math.floor(depth_src.shape[0]/box_p)
    num_box_w = math.floor(depth_src.shape[1]/box_p)
    max_h = depth_src.shape[0] - box_p
    max_w = depth_src.shape[1] - box_p
    _loss = torch.tensor(0.0,device='cuda')
    n_corr = int(p_corr * num_box_h * num_box_w)
    x_0 = torch.randint(0, max_h, size=(n_corr,), device = 'cuda')
    y_0 = torch.randint(0, max_w, size=(n_corr,), device = 'cuda')
    x_1 = x_0 + box_p
    y_1 = y_0 + box_p
    _loss = torch.tensor(0.0,device='cuda')
    for i in range(len(x_0)):
        _loss += pearson_depth_loss(depth_src[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1), depth_target[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1))
    return _loss/n_corr