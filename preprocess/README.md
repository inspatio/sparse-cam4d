# SparseCam4D — Preprocessing

This directory is the `preprocess/` module of the SparseCam4D project. It takes N synchronized fixed-camera videos and produces camera parameters, pseudo views, depth maps, and a time-tagged point cloud for 4DGS training.

## Outputs

| File / Dir | Description |
|---|---|
| `preprocess/time_XXXX/diffusion/` | Pseudo-view images (VGGT + ViewCrafter) |
| `sfm_transforms_extend.json` | All-view all-timestamp camera intrinsics/extrinsics |
| `depth/{cam}/{cam}_{t:04d}_depth.npy` | Per-camera per-frame metric depth maps |
| `vc_roma_sfm_300.ply` | Time-tagged point cloud (SfM + RoMa, optional) |

---

## 1. Clone & Submodule Setup

```bash
git clone <this-repo>
cd preprocess
git submodule update --init --recursive
```

This clones four submodules under `submodules/`:

- `VGGT` — camera pose estimation and point cloud rendering
- `RoMa` — dense image matching for depth alignment and point cloud merging
- `FlowFormerPlusPlus` — optical flow for motion mask generation
- `VideoDepthAnything` — video depth estimation

`submodules/ViewCrafter/` is a local copy (not a git submodule) — it is already present in the repository.

### 1.1 Model Checkpoints

The pipeline requires the following pretrained checkpoints. Download each to any local path and set the corresponding key in your config file.

| Config key | File | Source |
|---|---|---|
| `vggt_model_path` | `model.pt` | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| `viewcrafter_ckpt_path` | `model_sparse.ckpt` | [ViewCrafter project page](https://github.com/drexubery/ViewCrafter) |
| `roma_indoor_ckpt` | `roma_indoor.pth` | [Parskatt/RoMa releases](https://github.com/Parskatt/RoMa/releases) |
| `dinov2_ckpt` | `dinov2_vitl14_pretrain.pth` | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) — or set `HF_HOME` to use a local HuggingFace cache |
| `flowformer_ckpt` | `sintel.pth` | [XiaoyuShi97/FlowFormerPlusPlus](https://github.com/XiaoyuShi97/FlowFormerPlusPlus) |
| `vda_ckpt` | `video_depth_anything_vitl.pth` | [DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) |

---

## 2. Python Environment

```bash
# Other dependencies
pip install -r requirements.txt

# pytorch3d (required for ViewCrafter point cloud rendering)
# Option A — conda (recommended if available for your CUDA version):
conda install pytorch3d -c pytorch3d
# Option B — from source:
pip install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

# COLMAP (required for SfM stage)
conda install -c conda-forge colmap
```

> `timm==0.6.7` is pinned because FlowFormerPlusPlus uses APIs removed in `timm>=0.9`.

Each submodule may have additional dependencies — refer to the `README` in each submodule directory for details.

---

## 3. Config File

Copy and edit one of the example configs in `configs/`:

```yaml
# Input
cam_root: output/my_scene/images    # directory containing <cam>_<time>.png frames
train_cams: [cam01, cam05, cam10]   # camera names to use (order matters for interpolation)

# Output
output_root: output/my_scene

# Pipeline stages
run_pseudo_views: true
run_sfm: true
run_transforms: true
run_roma_pcd: false    # optional: foreground point cloud merging
run_video_depth: false # optional: metric depth estimation

# Model weights
vggt_model_path: /path/to/vggt/model.pt
viewcrafter_ckpt_path: /path/to/model_sparse.ckpt
roma_indoor_ckpt: /path/to/roma_indoor.pth
dinov2_ckpt: /path/to/dinov2_vitl14_pretrain.pth
flowformer_ckpt: /path/to/sintel.pth

# Runtime
device: cuda
gpu_id: 0
video_length: 25
```

See `configs/balloon1.yaml` for a full example with all options.

---

## 4. Running the Pipeline

```bash
CUDA_VISIBLE_DEVICES=0 python preprocess.py --config configs/balloon1.yaml
```

Or from Python:

```python
from preprocess import preprocess
preprocess("configs/balloon1.yaml")
```

### Pipeline Stages

| Stage | Flag | Description |
|---|---|---|
| 1 | `run_pseudo_views` | VGGT pose estimation → point cloud rendering → ViewCrafter pseudo-view generation |
| 2 | `run_sfm` | COLMAP SfM on time-0 pseudo views |
| 3 | `run_transforms` | Convert SfM poses to JSON; extend to all timestamps |
| 4 | `run_roma_pcd` | RoMa-based foreground/background point cloud merging (optional) |
| 5 | `run_video_depth` | VideoDepthAnything depth estimation + RoMa metric alignment (optional) |

### Input Data Format

Download the following datasets

- [Neural 3D Video Dataset](https://github.com/facebookresearch/Neural_3D_Video)
- [Technicolor](https://github.com/facebookresearch/hyperreel)

Then extract frames from raw videos each scene:

```
DATASET/SCENE/images
├── cam00_0000.png
├── cam00_0001.png
├── ...
├── cam05_0000.png
└── ...
```

Frames are named `{cam}_{time:04d}.png`. The `train_cams` list in the config must match the camera name prefixes and must be ordered consistently (the order determines the interpolation sequence for ViewCrafter). The `cam_root` in the config should refer to `DATASET/SCENE/images`.

---

## 5. Notes

The main bottleneck in the preprocessing pipeline is ViewCrafter. You can replace it with a more efficient SOTA novel-view synthesis model.
