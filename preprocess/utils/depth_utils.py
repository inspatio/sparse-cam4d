"""Stage 6 (Optional): Align VideoDepthAnything dense depth with RoMa sparse metric depth."""
import os
import sys
import json
import glob
import numpy as np
import cv2
import imageio
import torch
import torch.nn as nn
from tqdm import tqdm
from loguru import logger

try:
    import OpenEXR
    import Imath
    _HAS_EXR = True
except ImportError:
    _HAS_EXR = False

# ── RoMa helpers (reused from roma_pcd_utils) ────────────────────────────────
_SUBMODULES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submodules")
_ROMA_PATH = os.path.join(_SUBMODULES, "RoMa")
if _ROMA_PATH not in sys.path:
    sys.path.insert(0, _ROMA_PATH)


# ── Depth I/O helpers ─────────────────────────────────────────────────────────

def get_vda_infer(infer_path: str, target_size: tuple = None) -> np.ndarray:
    """Load depth from a VideoDepthAnything EXR file (Z channel).

    Args:
        infer_path: path to .exr file.
        target_size: (H, W) to resize to, or None to keep original.

    Returns:
        depth array (H, W) float32.
    """
    assert _HAS_EXR, "OpenEXR not installed"
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    exr = OpenEXR.InputFile(infer_path)
    dw = exr.header()["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    raw = exr.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
    if target_size is not None and (depth.shape[0] != target_size[0] or depth.shape[1] != target_size[1]):
        depth = cv2.resize(depth, (target_size[1], target_size[0]))
    return depth


def depth2disparity(depth: np.ndarray) -> np.ndarray:
    disp = np.zeros_like(depth)
    m = depth > 0
    disp[m] = 1.0 / depth[m]
    return disp


# ── Depth alignment ───────────────────────────────────────────────────────────

def align_depth(
    ref: np.ndarray,
    conf: np.ndarray,
    inf: np.ndarray,
    min_conf_thr: int = 30,
    valid_mask: np.ndarray = None,
    save_path: str = None,
) -> np.ndarray:
    """Align VDA disparity to RoMa metric depth via affine least-squares in disparity space.

    Solves:  inf_disp * s + t ≈ 1/ref_depth  over confident pixels.
    Returns metric depth map.
    """
    if valid_mask is None:
        conf_thr = np.percentile(conf.ravel(), min_conf_thr)
        conf_thr = float(np.clip(conf_thr, 1e-5, 1.0))
        valid_mask = (ref > 1e-3) & (conf > conf_thr)

    assert valid_mask.sum() > 0, "No valid pixels for depth alignment"

    ref_disp = 1.0 / (ref[valid_mask].reshape(-1, 1).astype(np.float64) + 1e-8)
    pred_disp = inf[valid_mask].reshape(-1, 1).astype(np.float64)
    A = np.concatenate([pred_disp, np.ones_like(pred_disp)], axis=-1)
    s, t = np.linalg.lstsq(A, ref_disp, rcond=None)[0]

    aligned = s * inf + t
    aligned = np.clip(aligned, 1e-3, None)
    aligned = depth2disparity(aligned)
    aligned = np.clip(aligned, 1e-3, None)
    assert not np.isnan(aligned).any(), "NaN in aligned depth"

    if save_path:
        np.save(save_path, aligned)
    return aligned


# ── RoMa sparse depth projection ─────────────────────────────────────────────

def _parse_transform(data: dict):
    K = np.array([
        [data["fx"], 0, data["cx"]],
        [0, data["fy"], data["cy"]],
        [0, 0, 1],
    ])
    c2w = np.array(data["transform_matrix"])[:3]
    size = (data["height"], data["width"])
    return K, c2w, data["file_path"], size


def _c2w_to_w2c(c2w):
    h = np.vstack([c2w, [0, 0, 0, 1]])
    return np.linalg.inv(h)[:3]


def _scale_K(K, orig_size, target_size):
    oh, ow = orig_size
    th, tw = target_size
    return np.diag([tw / ow, th / oh, 1.0]) @ K


def get_roma_sparse_depth(
    output_root: str,
    target_time: int,
    transforms_data: list,
    roma_model,
) -> tuple:
    """Run RoMa on all adjacent camera pairs at target_time, project 3D points → sparse depth.

    Returns:
        ref_depths: (N, H, W) float32
        confs: (N, H, W) uint8
        (H, W): output resolution
    """
    ts_workspace = os.path.join(output_root, "preprocess", f"time_{target_time:04d}")
    N = len(transforms_data)
    device = next(roma_model.parameters()).device
    H_r, W_r = roma_model.get_output_resolution()

    from romatch import roma_indoor  # ensure import available
    from PIL import Image as PILImage

    all_pts = []
    for i in range(N - 1):
        K1, c2w1, p1, s1 = _parse_transform(transforms_data[i])
        K2, c2w2, p2, s2 = _parse_transform(transforms_data[i + 1])
        fp1 = os.path.join(output_root, p1)
        fp2 = os.path.join(output_root, p2)

        warp, cert = roma_model.match(fp1, fp2, device=device)
        matches, _ = roma_model.sample(warp, cert)
        kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_r, W_r, H_r, W_r)
        border = 10
        m = (
            (kpts1[:, 0] > border) & (kpts1[:, 0] < W_r - border) &
            (kpts1[:, 1] > border) & (kpts1[:, 1] < H_r - border) &
            (kpts2[:, 0] > border) & (kpts2[:, 0] < W_r - border) &
            (kpts2[:, 1] > border) & (kpts2[:, 1] < H_r - border)
        )
        kpts1n = kpts1[m].cpu().numpy()
        kpts2n = kpts2[m].cpu().numpy()

        K1s = _scale_K(K1, s1, (H_r, W_r))
        K2s = _scale_K(K2, s2, (H_r, W_r))
        P1 = K1s @ _c2w_to_w2c(c2w1)
        P2 = K2s @ _c2w_to_w2c(c2w2)
        pts4d = cv2.triangulatePoints(
            P1, P2, kpts1n.T.astype(np.float32), kpts2n.T.astype(np.float32)
        )
        pts3d = (pts4d[:3] / pts4d[3:]).T
        all_pts.append(pts3d)

    if not all_pts:
        raise RuntimeError("RoMa returned no matches")
    pts3d_all = np.concatenate(all_pts, axis=0)

    # Project points to each cam's image plane
    N_pts = pts3d_all.shape[0]
    pts_h = np.concatenate([pts3d_all, np.ones((N_pts, 1))], axis=1)  # (N,4)

    ref_depths, confs = [], []
    for i in range(N):
        K, c2w, _, size = _parse_transform(transforms_data[i])
        H_out, W_out = size
        P = K @ _c2w_to_w2c(c2w)
        proj = (P @ pts_h.T).T  # (N,3)
        z = proj[:, 2]
        uv = proj[:, :2] / proj[:, 2:3]
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        valid = (u >= 0) & (u < W_out) & (v >= 0) & (v < H_out) & (z > 0)
        ref_d = np.zeros((H_out, W_out), dtype=np.float32)
        conf = np.zeros((H_out, W_out), dtype=np.uint8)
        for ii in np.where(valid)[0]:
            uu, vv, zz = u[ii], v[ii], z[ii]
            if conf[vv, uu] == 0 or zz < ref_d[vv, uu]:
                ref_d[vv, uu] = zz
                conf[vv, uu] = 1
        ref_depths.append(ref_d)
        confs.append(conf)

    return np.stack(ref_depths), np.stack(confs), (size[0], size[1])


# ── VideoDepthAnything inference ──────────────────────────────────────────────

def run_vda_inference(
    output_root: str,
    cam_root: str,
    train_cams: list,
    vda_path: str,
    vda_submodule: str = None,
) -> str:
    """Convert per-cam images to video and run VideoDepthAnything inference.

    Returns:
        vda_path: directory containing {cam}_depths_exr/ folders.
    """
    if vda_submodule is None:
        vda_submodule = os.path.join(_SUBMODULES, "VideoDepthAnything")

    os.makedirs(vda_path, exist_ok=True)
    for cam in train_cams:
        cam_dir = os.path.join(cam_root, cam)
        if not os.path.isdir(cam_dir):
            cam_dir = cam_root  # flat layout: images directly in cam_root
        input_video = os.path.join(vda_path, f"{cam}.mp4")
        # Stitch frames to video
        os.system(
            f'ffmpeg -framerate 30 -pattern_type glob '
            f'-i "{cam_dir}/{cam}_*.png" '
            f'-c:v libx264 -pix_fmt yuv444p "{input_video}" -v quiet -y'
        )
        exr_dir = os.path.join(vda_path, f"{cam}_depths_exr")
        exr_files = glob.glob(os.path.join(exr_dir, "*.exr"))
        if exr_files:
            logger.info(f"VDA EXR already exists for {cam} ({len(exr_files)} frames), skipping")
            continue
        run_cmd = (
            f"cd {vda_submodule} && python run.py "
            f"--input_video {input_video} "
            f"--output_dir {vda_path} --max_res 1352 --save_exr"
        )
        logger.info(f"Running VideoDepthAnything for {cam}")
        os.system(run_cmd)
    return vda_path


# ── Depth visualization ───────────────────────────────────────────────────────

def vis_depth(depth_root: str, train_cams: list) -> None:
    """Visualize aligned depth maps as colored MP4 videos."""
    import matplotlib.cm as cm
    colormap = np.array(cm.get_cmap("inferno").colors)

    all_depths = {}
    for cam in train_cams:
        npy_files = sorted(glob.glob(os.path.join(depth_root, cam, "*.npy")))
        all_depths[cam] = [np.load(p, allow_pickle=True).squeeze() for p in npy_files]

    # Global normalization across all cams
    d_min = min(d.min() for depths in all_depths.values() for d in depths)
    d_max = max(d.max() for depths in all_depths.values() for d in depths)

    for cam in train_cams:
        depths_arr = np.stack(all_depths[cam], axis=0)
        normed = np.clip((depths_arr - d_min) / (d_max - d_min + 1e-8), 0, 1)
        normed_u8 = (normed * 255).astype(np.uint8)
        vis = (colormap[normed_u8] * 255).astype(np.uint8)

        save_path = os.path.join(depth_root, f"{cam}.mp4")
        writer = imageio.get_writer(
            save_path, fps=30, codec="libx264",
            ffmpeg_log_level="error", pixelformat="yuv444p",
            ffmpeg_params=["-crf", "10"],
        )
        for frame in tqdm(vis, desc=f"Writing {cam}.mp4"):
            writer.append_data(frame)
        writer.close()
        logger.info(f"Saved depth video → {save_path}")


# ── Main Stage 6 entry point ──────────────────────────────────────────────────

def run_video_depth(
    output_root: str,
    cam_root: str,
    train_cams: list,
    Nframes: int,
    transforms_fn: str,
    roma_model,
    vda_path: str = None,
    video_len: int = 25,
    skip: int = 1,
    depth_folder: str = "depth",
):
    """Stage 6: align VideoDepthAnything dense depth with RoMa sparse metric depth.

    Steps:
        A. Run VDA inference (if vda_path is None or missing EXR files).
        B. Per-frame: get_roma_sparse_depth → align_depth → save .npy.
        C. vis_depth: generate {cam}.mp4 visualization.
    """
    depth_root = os.path.join(output_root, depth_folder)
    for cam in train_cams:
        os.makedirs(os.path.join(depth_root, cam), exist_ok=True)

    # ── A. VDA inference ──────────────────────────────────────────────────────
    if vda_path is None:
        vda_path = os.path.join(depth_root, "vda")
    vda_path = run_vda_inference(output_root, cam_root, train_cams, vda_path)

    # ── Load anchor camera list ───────────────────────────────────────────────
    t0_workspace = os.path.join(output_root, "preprocess", "time_0000")
    json_path = os.path.join(t0_workspace, transforms_fn)
    with open(json_path) as f:
        all_transforms = json.load(f)
    all_transforms = sorted(all_transforms, key=lambda x: x["id"])
    stride = (video_len - 1) // skip
    transforms_data = all_transforms[::stride][:len(train_cams)]

    # ── B. Per-frame alignment ────────────────────────────────────────────────
    for t in tqdm(range(Nframes), desc="Aligning depth"):
        # Update file_paths to current timestamp
        for i, cam in enumerate(train_cams):
            transforms_data[i]["file_path"] = os.path.join("images", f"{cam}_{t:04d}.png")

        refs, confs, (H, W) = get_roma_sparse_depth(
            output_root, t, transforms_data, roma_model
        )

        vda_paths = [
            os.path.join(vda_path, f"{cam}_depths_exr", f"frame_{t:05d}.exr")
            for cam in train_cams
        ]
        infers_raw = [get_vda_infer(p, target_size=(H, W)) for p in vda_paths]
        infers = np.stack(infers_raw, axis=0) / 100.0  # VDA scale factor

        for i, cam in enumerate(train_cams):
            save_path = os.path.join(depth_root, cam, f"{cam}_{t:04d}_depth.npy")
            align_depth(refs[i], confs[i], infers[i], save_path=save_path)

    # ── C. Visualization ──────────────────────────────────────────────────────
    vis_depth(depth_root, train_cams)
