# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

4D Dynamic Gaussian Splatting with Distortion Fields — a research implementation that extends 3D Gaussian Splatting to dynamic/temporal scenes using learnable HexPlane-based deformation networks. The codebase is Python + CUDA (PyTorch 2.1.0, CUDA 11.8).

## Environment Setup

```bash
conda env create --file environment.yml
conda activate 4ddistort
```

Custom CUDA extensions must be built before training:
```bash
cd diff-gaussian-rasterization && pip install -e .
cd simple-knn && pip install -e .
```

## Key Commands

**Training:**
```bash
python train.py --config configs/nvidia/balloon1.yaml
```

**Rendering (after training):**
```bash
python render_hexplane.py --config configs/nvidia/balloon1.yaml --skip_train --iteration 30000
python render_hexplane_spiral.py --config configs/nvidia/balloon1.yaml --skip_train --iteration 30000
```

**Evaluation metrics:**
```bash
python metrics.py --model_path <output_dir>
```

**Shell scripts for batch runs:**
- `train.sh` — basic training
- `ablation.sh` — ablation studies
- `render_opt.sh` — rendering with pose optimization

## Architecture

### Data Flow

```
train.py
  ├── Scene (scene/__init__.py) — loads dataset, initializes cameras & Gaussians
  │   ├── dataset_readers.py — supports Blender JSON, COLMAP, custom JSON formats
  │   └── MixGaussianModel — 4D Gaussian representation (x, y, z, t)
  ├── Deformation field (scene/deformation.py + scene/hexplane.py)
  │   └── HexPlaneField — 6-plane space-time decomposition with MLP decoder
  ├── gaussian_renderer/__init__.py — differentiable_render_with_hexplane()
  │   ├── applies distortion field transformations
  │   ├── computes 2D optical flow for temporal consistency
  │   └── diff-gaussian-rasterization/ — custom CUDA rasterizer
  └── loss_utils.py — L1 + SSIM + LPIPS + depth + motion smoothness
```

### Core Modules

**`scene/mix_gaussian_model.py`** — Central model: 4D Gaussians with optional 4D rotations (`rot_4d`), temporal SH features, and integration hooks for the deformation network.

**`scene/deformation.py`** — Deformation/distortion network: MLP that transforms Gaussian parameters (position, scale, rotation, opacity) using features from HexPlane. Supports FiLM modulation for temporal control and static-region detection.

**`scene/hexplane.py`** — HexPlaneField: decomposes 5D space-time (x,y,z,t + optional extra) into 6 pairs of 2D planes. Multi-resolution grids with trilinear interpolation.

**`gaussian_renderer/__init__.py`** — Main rendering entry point. `differentiable_render_with_hexplane()` handles camera pose application, distortion field query, and CUDA rasterization dispatch.

**`arguments/__init__.py`** — All hyperparameter groups: `ModelParams`, `PipelineParams`, `OptimizationParams`, `ModelHiddenParams`. Config YAML values override these defaults via OmegaConf.

### Configuration System

All experiments use YAML configs in `configs/` (datasets: `nvidia/`, `technicolor/`, `n3v/`, `ablation/`). Key parameters:

```yaml
gaussian_dim: 4          # 4 = 4D Gaussians (x,y,z,t)
rot_4d: True             # enable 4D rotations
num_pts: 300_000         # initial point count

ModelHiddenParams:
  train_distortion: True
  kplanes_config:
    resolution: [64, 64, 64, 101, 25]  # x,y,z,t,extra grid resolution
  apply_film_modulate: True

OptimizationParams:
  iterations: 30_000
  cam_optim_from_iter: 1000   # when pose optimization starts
  lambda_lpips: 0.2
  lambda_depth: 1.0
```

### CUDA Extension

`diff-gaussian-rasterization/` contains custom CUDA kernels for differentiable Gaussian splatting. The rasterizer outputs rendered image, depth, flow, and covariance. It must be compiled before use (`pip install -e .`).

## Output Structure

Training outputs go to `output/<experiment_name>/`:
- `point_cloud/iteration_N/` — saved Gaussian `.ply` files
- `cameras.json` — camera parameters
- `cfg_args` — saved config
- TensorBoard logs in the output directory
