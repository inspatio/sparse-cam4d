"""Stage 2a: VGGT model wrapper — load model, run inference, interpolate poses, render point cloud."""
import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF
from loguru import logger

from utils import load_submodule  # noqa: E402

_SUBMODULES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submodules")
_VGGT_PATH = os.path.join(_SUBMODULES, "VGGT")
if _VGGT_PATH not in sys.path:
    sys.path.insert(0, _VGGT_PATH)

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

_pvd = load_submodule("pvd_utils", os.path.join(_SUBMODULES, "ViewCrafter", "utils", "pvd_utils.py"))
to_numpy = _pvd.to_numpy
generate_traj_interp = _pvd.generate_traj_interp
setup_renderer = _pvd.setup_renderer
render_pcd = _pvd.render_pcd
save_render = _pvd.save_render
save_video = _pvd.save_video
import torch.nn.functional as F
import trimesh


def build_vggt_model(model_path: str, device: str = "cuda") -> VGGT:
    """Load VGGT model from checkpoint."""
    model = VGGT()
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)
    model = model.to(device).eval()
    logger.info(f"Loaded VGGT model from {model_path}")
    return model


def load_filelists(root: str, target_cams: list):
    """Build per-camera and per-timestamp file lists.

    Returns:
        ts_filelists: {timestamp_str: [file_paths]}
        cam_filelists: {cam_name: [file_paths]}  (absolute paths)
    """
    cam_files = []
    cam_filelists = {}

    for cam in target_cams:
        cam_dir = os.path.join(root, cam)
        if os.path.isdir(cam_dir):
            files = sorted(
                os.path.join(cam, f)
                for f in os.listdir(cam_dir) if f.endswith(".png")
            )
            cam_filelists[cam] = [os.path.join(root, f) for f in files]
        else:
            # Flat layout: images named <cam>_<time>.png directly under root/images/
            images_dir = os.path.join(root, "images")
            if not os.path.isdir(images_dir):
                images_dir = root
            files = sorted(
                os.path.join(images_dir, f)
                for f in os.listdir(images_dir)
                if f.startswith(f"{cam}_") and f.endswith(".png")
            )
            cam_filelists[cam] = files

        assert cam_filelists[cam], f"No images found for cam {cam} under {root}"
        cam_files.extend(cam_filelists[cam])

    ts_filelists = {}
    for fpath in cam_files:
        ts = os.path.basename(fpath).split(".")[0].split("_")[1]
        ts_filelists.setdefault(ts, []).append(fpath)

    return ts_filelists, cam_filelists


def _load_and_preprocess_images(filelists: list, device: str = "cuda"):
    """Load and resize images: width=518, height=14x.

    Returns:
        preprocessed_images: (N, C, H_small, W_small) tensor on device
        upsample_images: (N, H_orig, W_orig, C) tensor on device
    """
    to_tensor = TF.ToTensor()
    preprocessed, upsample = [], []
    target_width = 518

    for fpath in filelists:
        img = Image.open(fpath)
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        w, h = img.size

        resize_height = round(h * (target_width / w) / 14) * 14
        small = img.resize((target_width, resize_height), Image.Resampling.BICUBIC)
        preprocessed.append(to_tensor(small))
        upsample.append(to_tensor(img))

    preprocessed = torch.stack(preprocessed).to(device)          # (N, C, H, W)
    upsample = torch.stack(upsample).to(device).permute(0, 2, 3, 1).detach()  # (N, H, W, C)
    return preprocessed, upsample


def run_vggt(
    workspace: str,
    filelist: list,
    model: VGGT,
    known_extri_intri: dict = None,
    interp_pose: bool = True,
    pts_render_dir: str = "pts_render",
    video_length: int = 25,
    device: str = "cuda",
):
    """Run VGGT prediction + optional pose interpolation + point cloud rendering.

    Returns:
        (None, known_extri_intri): always returns updated camera params dict.
    """
    os.makedirs(workspace, exist_ok=True)
    logger.info(f"[VGGT] workspace={workspace}, {len(filelist)} images")

    imgs, upsample_images = _load_and_preprocess_images(filelist, device=device)

    # ── Predict ──────────────────────────────────────────────────────────────
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(imgs)

    # ── Extrinsics / Intrinsics ───────────────────────────────────────────────
    if known_extri_intri is None:
        pose_enc = predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
        known_extri_intri = {
            "extrinsic": extrinsic.tolist(),
            "intrinsic": intrinsic.tolist(),
        }
    else:
        extrinsic = torch.tensor(known_extri_intri["extrinsic"]).to(device)
        intrinsic = torch.tensor(known_extri_intri["intrinsic"]).to(device)

    depth_map = predictions["depth"]
    point_map = unproject_depth_map_to_point_map(
        depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
    )
    pts3d = torch.tensor(point_map).float()
    extrinsic = closed_form_inverse_se3(extrinsic.squeeze().detach())
    intrinsic = intrinsic.squeeze().detach()

    images = predictions["images"].squeeze().permute(0, 2, 3, 1).detach()  # (N, H, W, C)
    depths = depth_map.squeeze().detach()
    confs = predictions["depth_conf"].squeeze().detach()

    # ── Interpolate poses & render point cloud ────────────────────────────────
    if interp_pose:
        logger.info(f"[VGGT] Interpolating poses and rendering point cloud to {pts_render_dir}")
        _interp_and_render(
            workspace, extrinsic, intrinsic, images, upsample_images, pts3d,
            depths, confs, pts_render_dir=pts_render_dir, video_length=video_length
        )

    return None, known_extri_intri


def _interp_and_render(
    workspace, extrinsic, intrinsic, images, upsample_images,
    pts3d, depths, confs, pts_render_dir="pts_render", video_length=25
):
    H, W = images[0].shape[:2]
    focals = torch.tensor([[intrinsic[i, 0, 0], intrinsic[i, 1, 1]] for i in range(intrinsic.shape[0])])
    pps = torch.tensor([[intrinsic[i, 0, 2], intrinsic[i, 1, 2]] for i in range(intrinsic.shape[0])])

    camera_traj, num_views, _ = generate_traj_interp(
        extrinsic, H, W, focals, pps, video_length, mode="vggt", viz_traj=False, save_dir=workspace
    )

    # Render
    render_setup = setup_renderer(camera_traj, image_size=(H, W))
    renderer = render_setup["renderer"]
    pcd_pts = [i.detach().reshape(-1, 3) for i in pts3d]
    render_results, _ = render_pcd(pcd_pts, images, None, num_views, renderer, "cuda", nbv=False)

    # Upsample to original resolution
    uh, uw = upsample_images[0].shape[:2]
    render_results = F.interpolate(
        render_results.permute(0, 3, 1, 2), size=(uh, uw), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)

    # Replace anchor frames with original images
    for i in range(len(images)):
        render_results[i * (video_length - 1)] = upsample_images[i]

    save_render(render_results, os.path.join(workspace, pts_render_dir))
    save_video(render_results, os.path.join(workspace, "pts_render.mp4"))
