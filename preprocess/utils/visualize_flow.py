"""FlowFormerPlusPlus wrapper for optical flow computation.

Copied from submodules/FlowFormerPlusPlus/visualize_flow.py and adapted:
- build_model() accepts an explicit ckpt_path parameter instead of relying on
  the hardcoded path in configs/submissions.py, so the submodule is unmodified.
- All other logic is identical to the submodule version.
"""
import os
import sys
import math
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

_FF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submodules", "FlowFormerPlusPlus")
if _FF_PATH not in sys.path:
    sys.path.insert(0, _FF_PATH)
# FlowFormerPlusPlus uses relative sys.path.append('core'); patch it to absolute
_FF_CORE = os.path.join(_FF_PATH, "core")
if _FF_CORE not in sys.path:
    sys.path.insert(0, _FF_CORE)

from configs.submissions import get_cfg          # noqa: E402
from core.utils.misc import process_cfg          # noqa: E402
from core.utils import frame_utils               # noqa: E402
from core.FlowFormer import build_flowformer     # noqa: E402
from core.utils.utils import InputPadder         # noqa: E402

TRAIN_SIZE = [432, 960]


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
        raise ValueError(
            f"Overlap should be less than size of patch (got {min_overlap}"
            f"for patch size {patch_size}).")
    if image_shape[0] == TRAIN_SIZE[0]:
        hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
    else:
        hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
    if image_shape[1] == TRAIN_SIZE[1]:
        ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
    else:
        ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])
    return patch_weights


def compute_flow(model, image1, image2, weights=None):
    image_size = image1.shape[1:]
    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_pre, _ = model(image1, image2)
        flow_pre = padder.unpad(flow_pre)
    else:
        flows = 0
        flow_count = 0
        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)
        flow_pre = flows / flow_count

    return flow_pre


def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1]
    scale = scale0 if scale0 > scale1 else scale1
    return (int(image_size[1] * scale), int(image_size[0] * scale))


def prepare_image(root_dir, fn1, fn2, keep_size):
    image1 = frame_utils.read_gen(osp.join(root_dir, fn1))
    image2 = frame_utils.read_gen(osp.join(root_dir, fn2))
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    ori_size = image1.shape[0:2]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        inference_size = (dsize[1], dsize[0])
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    else:
        inference_size = ori_size
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
    return image1, image2, ori_size, inference_size


def build_model(ckpt_path: str):
    """Load FlowFormerPlusPlus model from an explicit checkpoint path."""
    cfg = get_cfg()
    cfg.model = ckpt_path          # override path from configs/submissions.py
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))
    model.cuda()
    model.eval()
    return model


def compute_optical_flow(model, root_dir, fn1, fn2, keep_size=False):
    image1, image2, ori_size, inference_size = prepare_image(root_dir, fn1, fn2, keep_size)
    flow = compute_flow(model, image1, image2, None)
    flow = F.interpolate(flow, size=ori_size, mode='bilinear', align_corners=True)
    flow[:, 0] = flow[:, 0] * ori_size[-1] / inference_size[-1]
    flow[:, 1] = flow[:, 1] * ori_size[-2] / inference_size[-2]
    flow = flow[0].permute(1, 2, 0).detach().cpu().numpy()
    return flow
