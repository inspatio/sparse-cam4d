"""SparseCam4D preprocessing pipeline — main entry point.

Usage:
    python preprocess.py --config configs/balloon1.yaml
    # or programmatically:
    from preprocess import preprocess
    preprocess("configs/balloon1.yaml")
"""
import os
import sys
import argparse
from omegaconf import OmegaConf
from loguru import logger

# Ensure the project root is in sys.path so `utils.*` resolves correctly
# regardless of the working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _setup_gpu(gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def preprocess(config_file: str) -> None:
    """Run the full SparseCam4D preprocessing pipeline.

    Args:
        config_file: path to a YAML config file (see configs/balloon1.yaml for reference).
    """
    cfg = OmegaConf.load(config_file)
    _setup_gpu(cfg.get("gpu_id", 0))

    # ── Resolve paths ─────────────────────────────────────────────────────────
    here = os.path.dirname(os.path.abspath(__file__))
    cam_root = os.path.join(here, cfg.cam_root) if not os.path.isabs(cfg.cam_root) else cfg.cam_root
    output_root = os.path.join(here, cfg.output_root) if not os.path.isabs(cfg.output_root) else cfg.output_root
    os.makedirs(output_root, exist_ok=True)

    train_cams: list = list(cfg.train_cams)
    diff_folder: str = cfg.get("diff_folder_name", "diffusion")
    sparse_folder: str = cfg.get("sparse_folder_name", "sparse")
    depth_folder: str = cfg.get("depth_folder_name", "depth")
    video_length: int = cfg.get("video_length", 25)
    skip: int = cfg.get("skip", 1)
    device: str = cfg.get("device", "cuda")

    # workspace = e.g. "preprocess/time_0000" relative to output_root
    t0_workspace = os.path.join(output_root, cfg.workspace)

    # ── Build file lists (needed by stages 2, 4, 5, 6) ───────────────────────
    from utils.vggt_utils import load_filelists
    ts_filelists, cam_filelists = load_filelists(cam_root, train_cams)
    timestamps = sorted(ts_filelists.keys())
    Nframes = len(timestamps)
    logger.info(f"Found {Nframes} timestamps, {len(train_cams)} train cams")

    # ── Stage 2: Pseudo-view generation ───────────────────────────────────────
    if cfg.get("run_pseudo_views", True):
        logger.info("=== Stage 2: VGGT + ViewCrafter pseudo-view generation ===")
        from utils.vggt_utils import build_vggt_model, run_vggt
        from utils.viewcrafter_utils import build_viewcrafter_model, run_viewcrafter

        recon_model = build_vggt_model(cfg.vggt_model_path, device=device)
        vc_model = build_viewcrafter_model(
            ckpt_path=cfg.get("viewcrafter_ckpt_path", None),
            config_path=cfg.get("viewcrafter_config_path", None),
            device=f"{device}:0" if ":" not in device else device,
        )

        known_extri_intri = None
        for ts in timestamps:
            ts_workspace = t0_workspace.replace("0000", ts)
            os.makedirs(ts_workspace, exist_ok=True)

            filelist = [
                os.path.join(cam_root, f"{cam}_{ts}.png")
                if not os.path.isdir(os.path.join(cam_root, cam))
                else os.path.join(cam_root, cam, f"{cam}_{ts}.png")
                for cam in train_cams
            ]
            # Use flat image layout: cam_root/<cam>_<ts>.png
            filelist = []
            for cam in train_cams:
                cam_dir = os.path.join(cam_root, cam)
                if os.path.isdir(cam_dir):
                    filelist.append(os.path.join(cam_dir, f"{cam}_{ts}.png"))
                else:
                    filelist.append(os.path.join(cam_root, f"{cam}_{ts}.png"))

            _, known_extri_intri = run_vggt(
                ts_workspace, filelist, model=recon_model,
                known_extri_intri=known_extri_intri,
                interp_pose=True, pts_render_dir="pts_render",
                video_length=video_length, device=device,
            )

            run_viewcrafter(
                ts_workspace,
                pts_render_dir="pts_render",
                diffusion_dir=diff_folder,
                viewcrafter_model=vc_model,
                img_ori=list(range(len(train_cams))),
                video_length=video_length,
            )

    # ── Stage 3: SfM at time=0 ────────────────────────────────────────────────
    if cfg.get("run_sfm", True):
        logger.info("=== Stage 3: SfM reconstruction at time=0 ===")
        from utils.sfm_utils import run_sfm
        run_sfm(
            workspace=t0_workspace,
            diffusion_dir=diff_folder,
            sparse_folder=sparse_folder,
            shared_intrinsics=False,
        )

    # ── Stage 4: Transform extension ──────────────────────────────────────────
    if cfg.get("run_transforms", True):
        logger.info("=== Stage 4: Extend transforms to all timestamps ===")
        from utils.transforms_utils import run_sfm_transforms_extend
        extend_path = os.path.join(output_root, "sfm_transforms_extend.json")
        run_sfm_transforms_extend(
            workspace=t0_workspace,
            extend_transforms_path=extend_path,
            cam_filelists=cam_filelists,
            diffusion_dir=diff_folder,
            sparse_folder=sparse_folder,
            video_len=video_length,
            depth_folder=depth_folder,
        )

    # ── Stage 5: RoMa point cloud merging (optional) ─────────────────────────
    if cfg.get("run_roma_pcd", False):
        logger.info("=== Stage 5: RoMa foreground point cloud merging ===")
        from utils.roma_pcd_utils import (
            generate_motion_masks, build_roma_model, run_roma_pcd
        )
        generate_motion_masks(output_root, cam_root, train_cams,
                              flowformer_ckpt=cfg.flowformer_ckpt)
        roma_model = build_roma_model(
            roma_indoor_ckpt=cfg.roma_indoor_ckpt,
            dinov2_ckpt=cfg.dinov2_ckpt,
            device=device,
        )
        # transforms_fn expected at time_0000/sfm_transforms.json (output of colmap2blender)
        run_roma_pcd(
            output_root=output_root,
            train_cams=train_cams,
            Nframes=Nframes,
            transforms_fn="sfm_transforms.json",
            roma_model=roma_model,
            video_len=video_length,
            skip=skip,
            sparse_folder=sparse_folder,
            ply_fn="vc_roma_sfm_300.ply",
        )

    # ── Stage 6: Video depth alignment (optional) ─────────────────────────────
    if cfg.get("run_video_depth", False):
        logger.info("=== Stage 6: Video depth alignment ===")
        from utils.depth_utils import run_video_depth

        # RoMa model may already be loaded; reuse if available, else build
        if not cfg.get("run_roma_pcd", False):
            from utils.roma_pcd_utils import build_roma_model
            roma_model = build_roma_model(
                roma_indoor_ckpt=cfg.roma_indoor_ckpt,
                dinov2_ckpt=cfg.dinov2_ckpt,
                device=device,
            )

        run_video_depth(
            output_root=output_root,
            cam_root=cam_root,
            train_cams=train_cams,
            Nframes=Nframes,
            transforms_fn="sfm_transforms.json",
            roma_model=roma_model,
            vda_path=cfg.get("vda_path", None),
            video_len=video_length,
            skip=skip,
            depth_folder=depth_folder,
        )

    logger.info("=== SparseCam4D preprocessing complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SparseCam4D Preprocessing Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    preprocess(args.config)
