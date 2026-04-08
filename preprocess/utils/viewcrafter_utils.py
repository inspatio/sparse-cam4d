"""Stage 2b: ViewCrafter wrapper — generate pseudo views per timestamp."""
import os
import sys
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from loguru import logger

from utils import load_submodule  # noqa: E402

_SUBMODULES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submodules")
_VC_PATH = os.path.join(_SUBMODULES, "ViewCrafter")

# ViewCrafter's internal imports use `from utils.pvd_utils` / `from utils.diffusion_utils`.
# We must ensure ViewCrafter/ is in sys.path *before* the project root so that
# `utils` resolves to ViewCrafter/utils/ (which has __init__.py) for those imports.
if _VC_PATH not in sys.path:
    sys.path.insert(0, _VC_PATH)

_pvd = load_submodule("pvd_utils", os.path.join(_VC_PATH, "utils", "pvd_utils.py"))
save_render = _pvd.save_render
save_video = _pvd.save_video

# Also pre-load diffusion_utils so viewcrafter.py's `from utils.diffusion_utils` resolves
load_submodule("utils.diffusion_utils", os.path.join(_VC_PATH, "utils", "diffusion_utils.py"))
load_submodule("utils.pvd_utils", os.path.join(_VC_PATH, "utils", "pvd_utils.py"))

from viewcrafter import ViewCrafter  # noqa: E402


def build_viewcrafter_model(
    ckpt_path: str = None,
    config_path: str = None,
    device: str = "cuda:0",
) -> ViewCrafter:
    """Instantiate ViewCrafter model (no save_dir yet; set per-workspace via set_save_dir)."""
    model = ViewCrafter(
        save_dir=None,
        mode="sparse",
        ckpt_path=ckpt_path,
        config_path=config_path,
        device=device,
    )
    logger.info("ViewCrafter model loaded")
    return model


def run_viewcrafter(
    workspace: str,
    pts_render_dir: str = "pts_render",
    diffusion_dir: str = "diffusion",
    viewcrafter_model: ViewCrafter = None,
    img_ori: list = None,
    video_length: int = 25,
):
    """Run ViewCrafter on rendered point cloud images to generate pseudo views.

    Reads sorted PNGs from workspace/pts_render_dir, generates diffusion results
    in video_length-frame clips (one per adjacent camera pair), saves to workspace/diffusion_dir.

    Args:
        workspace: directory containing pts_render_dir and where diffusion_dir will be created.
        pts_render_dir: subdir with rendered point cloud frames.
        diffusion_dir: subdir to save generated pseudo-view frames.
        viewcrafter_model: pre-loaded ViewCrafter instance.
        img_ori: list of anchor frame indices (one per train cam); default [0,1,...,N-1].
        video_length: frames per interpolation clip (default 25).
    """
    assert viewcrafter_model is not None, "Pass a pre-loaded ViewCrafter model"
    assert os.path.isdir(workspace), f"workspace does not exist: {workspace}"

    viewcrafter_model.set_save_dir(workspace)

    # Load rendered point cloud images
    render_files = sorted(
        f for f in os.listdir(os.path.join(workspace, pts_render_dir)) if f.endswith(".png")
    )
    render_results = []
    for fname in render_files:
        img = TF.to_tensor(
            Image.open(os.path.join(workspace, pts_render_dir, fname)).convert("RGB")
        )
        render_results.append(img)
    render_results = (
        torch.stack(render_results, dim=0).float().cuda().permute(0, 2, 3, 1)
    )  # (T, H, W, C)

    if img_ori is None:
        # One anchor per train cam: 0, video_length-1, 2*(video_length-1), ...
        n_clips = (len(render_files) - 1) // (video_length - 1)
        img_ori = list(range(n_clips + 1))

    logger.info(f"[ViewCrafter] Generating {len(img_ori)-1} clips at {workspace}")

    diffusion_results = []
    for i in range(len(img_ori) - 1):
        logger.info(f"[ViewCrafter] Clip {i+1}/{len(img_ori)-1}")
        start = i * (video_length - 1)
        end = start + video_length
        clip = viewcrafter_model.run(render_results[start:end], save=False)
        if i == 0:
            diffusion_results.append(clip)
        else:
            diffusion_results.append(clip[1:])  # skip duplicate anchor frame

    diffusion_results = torch.cat(diffusion_results, dim=0)  # (total_frames, H, W, C)

    # Replace anchor frames with original rendered images
    for i in range(len(img_ori)):
        diffusion_results[i * (video_length - 1)] = render_results[i * (video_length - 1)]

    save_render(diffusion_results, os.path.join(workspace, diffusion_dir))
    save_video(diffusion_results, os.path.join(workspace, "diffusion_vc.mp4"))
    logger.info(f"[ViewCrafter] Saved {len(diffusion_results)} frames to {diffusion_dir}")
