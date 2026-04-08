"""Stage 4: Convert SfM results to 4DGS JSON format and extend across all timestamps."""
import os
import sys
import json
import copy
import numpy as np
from tqdm import tqdm
from loguru import logger

from utils import load_submodule  # noqa: E402

_SUBMODULES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submodules")
_rwm = load_submodule("read_write_model", os.path.join(_SUBMODULES, "ViewCrafter", "utils", "read_write_model.py"))
read_model = _rwm.read_model
qvec2rotmat = _rwm.qvec2rotmat


def colmap2blender(
    workspace: str,
    timestamp: float = 0.0,
    diffusion_dir: str = "diffusion",
    sparse_folder: str = "sparse",
    transforms_fn: str = "sfm_transforms.json",
    video_len: int = 25,
) -> str:
    """Convert COLMAP sparse reconstruction to camera JSON (anchor at one timestamp).

    Returns path to the saved JSON file.
    """
    sfm_space = os.path.join(workspace, sparse_folder, "0")
    cameras, images, _ = read_model(sfm_space)
    N_interp = len(images)
    images = dict(sorted(images.items(), key=lambda item: item[0]))

    transforms = []
    for _, im in images.items():
        qvec = im.qvec
        tvec = im.tvec
        cam_id = im.camera_id
        image_name = im.name
        name_split = image_name.split(".")[0].split("_")
        file_path = os.path.join(diffusion_dir, name_split[0] + "_" + name_split[1] + ".png")
        frame_id = int(name_split[1])

        R = qvec2rotmat(qvec)
        T = np.array(tvec)
        w2c = np.eye(4)
        w2c[:3, 3] = T
        w2c[:3, :3] = R
        c2w = np.linalg.inv(w2c)

        intr = cameras[cam_id]
        width, height = intr.width, intr.height
        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = intr.params[0]
            cx, cy = intr.params[1], intr.params[2]
        elif intr.model == "PINHOLE":
            fx, fy = intr.params[0], intr.params[1]
            cx, cy = intr.params[2], intr.params[3]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {intr.model}")

        camera = {
            "id": frame_id,
            "file_path": file_path,
            "width": width,
            "height": height,
            "transform_matrix": c2w.tolist(),
            "position": c2w[:3, 3].tolist(),
            "rotation": c2w[:3, :3].tolist(),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "time_pseudo": (im.id - 1) / N_interp if frame_id % (video_len - 1) != 0 else 0,
            "time": timestamp,
            "is_true_image": 1 if frame_id % (video_len - 1) == 0 else 0,
        }
        transforms.append(camera)

    transforms_path = os.path.join(workspace, transforms_fn)
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=4)
    logger.info(f"colmap2blender: wrote {len(transforms)} cameras to {transforms_path}")
    return transforms_path


def extend_anchor_to_all_timestamps(
    anchor_transforms_path: str,
    extend_transforms_path: str,
    cam_filelists: dict,
    Nframes: int = -1,
    diffusion_dir: str = "diffusion",
    depth_folder: str = "depth",
) -> None:
    """Extend time-0 SfM camera poses to all timestamps and write final JSON.

    Real images get a depth_path field; pseudo images do not.

    Args:
        anchor_transforms_path: path to time-0 JSON (output of colmap2blender).
        extend_transforms_path: path to write the extended JSON.
        cam_filelists: {cam_name: [sorted abs file paths]} from load_filelists().
        Nframes: total timestamps; -1 = infer from cam_filelists.
        diffusion_dir: name of the pseudo-view subdirectory.
        depth_folder: name of the depth output directory (relative to output_root).
    """
    with open(anchor_transforms_path) as f:
        anchor_cam_infos = json.load(f)

    input_cams = list(cam_filelists.keys())
    total_ts = Nframes if Nframes > 0 else len(cam_filelists[input_cams[0]])

    anchor_real = [c for c in anchor_cam_infos if c["is_true_image"]]
    anchor_pseudo = [c for c in anchor_cam_infos if not c["is_true_image"]]
    logger.info(f"Extending {len(anchor_real)} real + {len(anchor_pseudo)} pseudo cams × {total_ts} timestamps")

    extend_cam_infos = []
    N_cam = 0
    total_iters = total_ts * len(anchor_cam_infos)
    progress = tqdm(total=total_iters, desc="Extending transforms")

    # ── Real cameras: one entry per (cam × timestamp) ─────────────────────────
    for cam_i, cam in enumerate(input_cams):
        filelist = sorted(cam_filelists[cam])
        for ts, fpath in enumerate(filelist):
            frame = os.path.basename(fpath)
            cam_info = copy.deepcopy(anchor_real[cam_i])
            cam_info["id"] = N_cam
            cam_info["file_path"] = os.path.join("images", frame)
            cam_info["depth_path"] = os.path.join(
                depth_folder, cam, frame.replace(".png", "_depth.npy")
            )
            cam_info["time_pseudo"] = 0.0
            cam_info["time"] = ts / total_ts
            extend_cam_infos.append(cam_info)
            progress.update(1)
        N_cam += 1

    # ── Pseudo cameras: one entry per (pseudo_view × timestamp) ──────────────
    for anchor_cam_info in anchor_pseudo:
        for ts in range(total_ts):
            cam_info = copy.deepcopy(anchor_cam_info)
            cam_info["id"] = N_cam
            cam_info["time"] = ts / total_ts
            # Update file_path to point to correct timestamp's diffusion dir
            # format: preprocess/time_XXXX/diffusion/frame_XXXXX.png
            base_fname = os.path.basename(cam_info["file_path"])
            cam_info["file_path"] = os.path.join(
                "preprocess", f"time_{ts:04d}", diffusion_dir, base_fname
            )
            extend_cam_infos.append(cam_info)
            progress.update(1)
        N_cam += 1

    progress.close()
    logger.info(f"Extended to {len(extend_cam_infos)} total camera entries")

    with open(extend_transforms_path, "w") as f:
        json.dump(extend_cam_infos, f, indent=4)
    logger.info(f"Saved extended transforms to {extend_transforms_path}")


def run_sfm_transforms_extend(
    workspace: str,
    extend_transforms_path: str,
    cam_filelists: dict,
    diffusion_dir: str = "diffusion",
    sparse_folder: str = "sparse",
    video_len: int = 25,
    depth_folder: str = "depth",
) -> None:
    """Full Stage 4: colmap2blender → extend_anchor_to_all_timestamps.

    Args:
        workspace: time_0000 workspace directory containing sparse_folder/.
        extend_transforms_path: output JSON path (e.g. output_root/sfm_transforms_extend.json).
        cam_filelists: {cam: [file_paths]} from vggt_utils.load_filelists().
        diffusion_dir: pseudo-view subdir name.
        sparse_folder: COLMAP sparse subdir name.
        video_len: frames per ViewCrafter clip.
        depth_folder: depth output directory name.
    """
    anchor_path = colmap2blender(
        workspace,
        timestamp=0.0,
        diffusion_dir=diffusion_dir,
        sparse_folder=sparse_folder,
        transforms_fn="sfm_transforms.json",
        video_len=video_len,
    )
    extend_anchor_to_all_timestamps(
        anchor_path, extend_transforms_path, cam_filelists,
        diffusion_dir=diffusion_dir, depth_folder=depth_folder,
    )
