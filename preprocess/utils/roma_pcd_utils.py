"""Stage 5 (Optional): RoMa-based foreground/background point cloud merging."""
import os
import sys
import json
import numpy as np
import cv2
import torch
from glob import glob
from tqdm import tqdm
from loguru import logger

# Add RoMa to path
_SUBMODULES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submodules")
_ROMA_PATH = os.path.join(_SUBMODULES, "RoMa")
if _ROMA_PATH not in sys.path:
    sys.path.insert(0, _ROMA_PATH)

# Compatibility shim for older FlowFormerPlusPlus with newer timm
try:
    import timm.models.helpers as _timm_helpers
    if not hasattr(_timm_helpers, "overlay_external_default_cfg"):
        _timm_helpers.overlay_external_default_cfg = lambda cfg, kwargs: kwargs
except ImportError:
    pass

from romatch import roma_indoor  # noqa: E402

from .sfm_utils import store_ply_with_time, load_ply  # noqa: E402


# ── RoMa model ────────────────────────────────────────────────────────────────

def build_roma_model(
    roma_indoor_ckpt: str,
    dinov2_ckpt: str,
    device: str = "cuda",
):
    """Load RoMa indoor model."""
    weights = torch.load(roma_indoor_ckpt, map_location=device)
    dinov2_weights = torch.load(dinov2_ckpt, map_location=device)
    model = roma_indoor(device=device, weights=weights, dinov2_weights=dinov2_weights)
    logger.info("RoMa model loaded")
    return model


# ── Camera parameter helpers ──────────────────────────────────────────────────

def _parse_transform(data: dict):
    K = np.array([
        [data["fx"], 0, data["cx"]],
        [0, data["fy"], data["cy"]],
        [0, 0, 1],
    ])
    c2w = np.array(data["transform_matrix"])[:3]  # (3, 4)
    size = (data["height"], data["width"])
    return K, c2w, data["file_path"], size


def _c2w_to_w2c(c2w):
    """Convert 3×4 c2w to 3×4 w2c."""
    c2w_h = np.vstack([c2w, [0, 0, 0, 1]])
    return np.linalg.inv(c2w_h)[:3]


def _scale_K(K, orig_size, target_size):
    oh, ow = orig_size
    th, tw = target_size
    S = np.diag([tw / ow, th / oh, 1.0])
    return S @ K


# ── RoMa matching + triangulation ─────────────────────────────────────────────

def run_roma(workspace: str, transforms_data: list, roma_model) -> tuple:
    """Match two views with RoMa and triangulate 3D points.

    Args:
        workspace: root dir; file_path in transforms_data is relative to this.
        transforms_data: list of exactly 2 camera dicts.
        roma_model: loaded RoMa model.

    Returns:
        (coords1, coords2, points3d, colors) — all numpy arrays.
    """
    assert len(transforms_data) == 2
    K1, c2w1, path1, size1 = _parse_transform(transforms_data[0])
    K2, c2w2, path2, size2 = _parse_transform(transforms_data[1])
    path1 = os.path.join(workspace, path1)
    path2 = os.path.join(workspace, path2)

    device = next(roma_model.parameters()).device
    H, W = roma_model.get_output_resolution()

    warp, certainty = roma_model.match(path1, path2, device=device)
    matches, _ = roma_model.sample(warp, certainty)
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H, W, H, W)

    border = 10
    mask = (
        (kpts1[:, 0] > border) & (kpts1[:, 0] < W - border) &
        (kpts1[:, 1] > border) & (kpts1[:, 1] < H - border) &
        (kpts2[:, 0] > border) & (kpts2[:, 0] < W - border) &
        (kpts2[:, 1] > border) & (kpts2[:, 1] < H - border)
    )
    kpts1 = kpts1[mask].cpu().numpy()
    kpts2 = kpts2[mask].cpu().numpy()

    K1s = _scale_K(K1, size1, (H, W))
    K2s = _scale_K(K2, size2, (H, W))
    P1 = K1s @ _c2w_to_w2c(c2w1)
    P2 = K2s @ _c2w_to_w2c(c2w2)

    pts4d = cv2.triangulatePoints(P1, P2, kpts1.T.astype(np.float32), kpts2.T.astype(np.float32))
    pts3d = (pts4d[:3] / pts4d[3:]).T  # (N, 3)

    # Color from average of both views
    from PIL import Image as PILImage
    imgA = np.array(PILImage.open(path1).resize((W, H))).astype(np.float32) / 255.0
    imgB = np.array(PILImage.open(path2).resize((W, H))).astype(np.float32) / 255.0
    c1 = kpts1.astype(int)
    c2 = kpts2.astype(int)
    colors = np.array([
        (imgA[v, u] + imgB[v2, u2]) / 2.0
        for (u, v), (u2, v2) in zip(c1, c2)
    ])

    # Filter outliers by distance from origin
    norms = np.linalg.norm(pts3d, axis=1)
    thr = np.percentile(norms, 95)
    valid = norms < thr
    return c1[valid], c2[valid], pts3d[valid], colors[valid]


# ── Motion mask generation ────────────────────────────────────────────────────

def generate_motion_masks(
    output_root: str,
    cam_root: str,
    train_cams: list,
    flowformer_ckpt: str,
    flow_threshold: float = 1.0,
):
    """Generate per-camera per-frame motion masks using FlowFormerPlusPlus + MaskRCNN.

    Output: output_root/motion_masks/{cam}/{cam}_{time:04d}.png
    """
    try:
        from utils.visualize_flow import build_model, compute_optical_flow
    except ImportError as e:
        raise ImportError(f"FlowFormerPlusPlus not available: {e}")

    import torchvision
    import skimage.morphology
    from PIL import Image as PILImage

    # Build models
    net_flow = build_model(flowformer_ckpt)

    net_maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()

    _MOVABLE_LABELS = {1, 2, 4, 8, 17, 18, 28, 36, 41}  # person, bicycle, motorcycle, truck, cat, dog, umbrella, snowboard, skateboard

    for cam in train_cams:
        cam_dir = os.path.join(cam_root, cam)
        if os.path.isdir(cam_dir):
            img_list = sorted(os.listdir(cam_dir))
            img_dir = cam_dir
        else:
            img_dir = cam_root
            img_list = sorted(
                os.path.basename(f) for f in glob(os.path.join(cam_root, f"{cam}_*.png"))
            )

        save_dir = os.path.join(output_root, "motion_masks", cam)
        os.makedirs(save_dir, exist_ok=True)

        if len(img_list) == len(os.listdir(save_dir)):
            logger.info(f"Motion masks for {cam} already exist, skipping")
            continue

        num_frames = len(img_list)
        logger.info(f"Generating motion masks for {cam}: {num_frames} frames")

        img0 = cv2.imread(os.path.join(img_dir, img_list[0]))
        H, W = img0.shape[:2]

        for i in tqdm(range(num_frames), desc=f"motion_mask/{cam}"):
            im_prev = img_list[max(0, i - 1)]
            im_ref = img_list[i]
            im_post = img_list[min(num_frames - 1, i + 1)]
            img_path = os.path.join(img_dir, im_ref)

            # Optical flow
            if i == 0:
                fwd = compute_optical_flow(net_flow, img_dir, im_ref, im_post)
                bwd = np.zeros_like(fwd)
            elif i == num_frames - 1:
                bwd = compute_optical_flow(net_flow, img_dir, im_ref, im_prev)
                fwd = np.zeros_like(bwd)
            else:
                fwd = compute_optical_flow(net_flow, img_dir, im_ref, im_post)
                bwd = compute_optical_flow(net_flow, img_dir, im_ref, im_prev)

            e_dist = np.maximum(np.linalg.norm(fwd, axis=2), np.linalg.norm(bwd, axis=2))
            flow_mask = skimage.morphology.binary_opening(e_dist > flow_threshold, skimage.morphology.disk(1))

            # MaskRCNN semantic mask
            img_pil = PILImage.open(img_path).resize((1024, 576))
            img_t = torchvision.transforms.functional.to_tensor(img_pil).cuda()
            with torch.no_grad():
                preds = net_maskrcnn([img_t])[0]
            sem_mask = torch.ones(576, 1024, dtype=torch.float32).cuda()
            for k in range(preds["masks"].size(0)):
                if preds["scores"][k] > 0.5 and preds["labels"][k].item() in _MOVABLE_LABELS:
                    sem_mask[preds["masks"][k, 0] > 0.5] = 0.0
            sem_mask = (sem_mask < 1e-3).cpu().numpy()
            sem_mask = cv2.resize(sem_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

            combined = skimage.morphology.dilation(
                flow_mask | sem_mask, skimage.morphology.disk(2)
            )
            img_name = os.path.basename(img_path)
            cv2.imwrite(
                os.path.join(save_dir, img_name),
                np.uint8(np.clip(combined, 0, 1) * 255),
            )


def _load_motion_masks(motion_folder: str, mask_paths: list, target_shape: tuple = ()) -> np.ndarray:
    """Load and optionally resize motion masks. Returns (N, H, W) bool array."""
    masks = []
    for p in mask_paths:
        m = cv2.imread(p)
        if len(target_shape) == 2:
            m = cv2.resize(m, target_shape, interpolation=cv2.INTER_NEAREST)
        masks.append(m[:, :, 0] > 0.1)
    return np.stack(masks)


# ── Point cloud merging ───────────────────────────────────────────────────────

def merge_fgd_pts(
    output_root: str,
    transforms_fn: str,
    train_cams: list,
    Nframes: int,
    roma_model,
    video_len: int = 25,
    skip: int = 1,
    sparse_folder: str = "sparse",
    ply_fn: str = "vc_roma_sfm_300.ply",
):
    """Merge foreground (dynamic) and background (static) point clouds using RoMa.

    Core logic:
    - For each frame pair <t, t+1>: RoMa matching → split into foreground/background by motion mask
    - Foreground: append per-frame with time = t/Nframes
    - Background: from t=0 only, with random time in [-0.1, 1.1]
    - SfM points from t=0: also append with random time

    Output: output_root/ply_fn
    """
    motion_mask_folder = os.path.join(output_root, "motion_masks")
    t0_workspace = os.path.join(output_root, "preprocess", "time_0000")
    json_path = os.path.join(t0_workspace, transforms_fn)

    # Load anchor camera list (one per train cam: stride = (video_len-1)//skip)
    with open(json_path) as f:
        all_transforms = json.load(f)
    all_transforms = sorted(all_transforms, key=lambda x: x["id"])
    stride = (video_len - 1) // skip
    transforms_data = all_transforms[::stride][:len(train_cams)]
    # Fix file_path to point at t=0 cam images
    for i, cam in enumerate(train_cams):
        transforms_data[i]["file_path"] = os.path.join("images", f"{cam}_0000.png")
    logger.info(f"Anchor cameras: {[d['file_path'] for d in transforms_data]}")

    H_r, W_r = roma_model.get_output_resolution()

    xyzs, rgbs, times = [], [], []
    static_xyz_list, static_rgb_list = [], []

    for t in tqdm(range(Nframes), desc="Merging foreground pts"):
        ts_workspace = os.path.join(output_root, "preprocess", f"time_{t:04d}")
        mask_paths = [
            os.path.join(motion_mask_folder, cam, f"{cam}_{t:04d}.png")
            for cam in train_cams
        ]
        motion_masks = _load_motion_masks(
            motion_mask_folder, mask_paths, target_shape=(W_r, H_r)
        )

        # Update file_path to current timestamp
        if t > 0:
            for i, cam in enumerate(train_cams):
                transforms_data[i]["file_path"] = os.path.join("images", f"{cam}_{t:04d}.png")

        xyz_t, rgb_t = [], []
        for i in range(len(transforms_data) - 1):
            c1, c2, pts3d, colors = run_roma(output_root, transforms_data[i:i + 2], roma_model)
            fgd = (motion_masks[i][c1[:, 1], c1[:, 0]] > 0) | (motion_masks[i + 1][c2[:, 1], c2[:, 0]] > 0)
            xyz_t.append(pts3d[fgd])
            rgb_t.append(colors[fgd])

            if t == 0:
                static_xyz_list.append(pts3d[~fgd])
                static_rgb_list.append(colors[~fgd])

        xyz_t = np.concatenate(xyz_t, axis=0)
        rgb_t = np.concatenate(rgb_t, axis=0)
        xyzs.append(xyz_t)
        rgbs.append(np.uint8(np.clip(rgb_t, 0, 1) * 255))
        times.append(np.ones(len(xyz_t), dtype=np.float64) * t / Nframes)

        if t == 0:
            # Static background from RoMa t=0
            s_xyz = np.concatenate(static_xyz_list, axis=0)
            s_rgb = np.concatenate(static_rgb_list, axis=0)
            xyzs.append(s_xyz)
            rgbs.append(np.uint8(np.clip(s_rgb, 0, 1) * 255))
            times.append(np.random.rand(len(s_xyz)) * 1.2 - 0.1)

            # SfM points from sparse/0/points3D.ply at t=0
            sfm_ply = os.path.join(t0_workspace, sparse_folder, "0", "points3D.ply")
            if os.path.exists(sfm_ply):
                sfm_xyz, sfm_rgb, _, _ = load_ply(sfm_ply)
                sfm_rgb_f = sfm_rgb.astype(np.float32) / 255.0
                xyzs.append(sfm_xyz)
                rgbs.append(np.uint8(np.clip(sfm_rgb_f, 0, 1) * 255))
                times.append(np.random.rand(len(sfm_xyz)) * 1.2 - 0.1)

    xyz_all = np.concatenate(xyzs, axis=0)
    rgb_all = np.concatenate(rgbs, axis=0)
    time_all = np.concatenate(times, axis=0)

    ply_path = os.path.join(output_root, ply_fn)
    store_ply_with_time(ply_path, xyz_all, rgb_all, time_all)
    logger.info(f"Saved {len(xyz_all)} points → {ply_path}  (time: {time_all.min():.2f}~{time_all.max():.2f})")


def run_roma_pcd(
    output_root: str,
    train_cams: list,
    Nframes: int,
    transforms_fn: str,
    roma_model,
    video_len: int = 25,
    skip: int = 1,
    sparse_folder: str = "sparse",
    ply_fn: str = "vc_roma_sfm_300.ply",
):
    """Stage 5 entry point."""
    merge_fgd_pts(
        output_root=output_root,
        transforms_fn=transforms_fn,
        train_cams=train_cams,
        Nframes=Nframes,
        roma_model=roma_model,
        video_len=video_len,
        skip=skip,
        sparse_folder=sparse_folder,
        ply_fn=ply_fn,
    )
