import sys
import os
from scene import Scene, MixGaussianModel
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import torchvision

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from tqdm import tqdm
from utils.image_utils import psnr, save_video
from utils.loss_utils import ssim, PerceptualLoss
from utils.general_utils import safe_state
from gaussian_renderer import differentiable_render_with_hexplane
from loguru import logger
import shutil
import json
from utils.general_utils import get_expon_lr_func

perceptual_loss = PerceptualLoss('alex', device=f'cuda:0')

def render_sets(dataset, pipe, load_iteration, skip_train, skip_test,
                gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d,
                target_cam, target_ts_train, target_ts_test, hyper=None, opt=None):
    dataset.transforms_file = os.path.join(dataset.model_path, "pose/sfm_transforms_to_ours.json")
    with open(dataset.transforms_file) as f:
        transforms = json.load(f)
        target_cam_train = transforms['train_cam']
        target_cam_test = transforms['test_cam']
    target_cam = {
        'train': target_cam_train,
        'test': target_cam_test
    }
    with torch.no_grad():
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]

        min_timestep = 1.0/dataset.Ntime/dataset.frame_ratio
        scene = Scene(dataset, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration, target_cam=target_cam,
                      target_time_train=[ts*min_timestep for ts in target_ts_train], target_time_test=[ts*min_timestep for ts in target_ts_test],
                      shuffle=False, cached_dataset=False)
        gaussians = MixGaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d,
                                     sh_degree_t=2 if pipe.eval_shfs_4d else 0, distortion_args=hyper, frame_ratio=dataset.frame_ratio, Ntime=dataset.Ntime)

        checkpoint = os.path.join(dataset.model_path, f"chkpnt{load_iteration}.pth")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, None, scene.getTrainDifferentiableCameras())
        logger.info(f"Loaded trained model from {dataset.model_path} at iteration {first_iter}")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
        differentiable_render_set_optimize(dataset.model_path, "train", load_iteration, scene.getTrainCameras(), scene.getTrainDifferentiableCameras(),
                                        gaussians, pipe, background, opt=opt)
    if not skip_test:
        differentiable_render_set_optimize(dataset.model_path, "test", load_iteration, scene.getTestCameras(), scene.getTestDifferentiableCameras(),
                                        gaussians, pipe, background, opt=opt)

def differentiable_render_set_optimize(model_path, name, iteration, views, differentiable_cams, gaussians, pipeline, background, opt=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_opt")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    if os.path.exists(render_path):
        shutil.rmtree(render_path)
    os.makedirs(render_path, exist_ok=True)
    if os.path.exists(gts_path):
        shutil.rmtree(gts_path)
    os.makedirs(gts_path, exist_ok=True)

    #! get camera list. {cam: [data_idx_list]}
    cam_idx_list = {}
    for idx in tqdm(range(len(views))):
        view = views.viewpoint_stack[idx]
        cam, timestamp = view.image_name.split('.')[0].split("_")
        if cam not in cam_idx_list:
            cam_idx_list[cam] = []
        if timestamp == '0000':
            cam_idx_list[cam].insert(0, idx)
        else:
            cam_idx_list[cam].append(idx)
    logger.info(f"Render cameras {cam_idx_list.keys()}")

    #! Freeze Gaussian Attributes
    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._t.requires_grad_(False)
    gaussians._scaling_t.requires_grad_(False)
    gaussians._rotation_r.requires_grad_(False)

    num_iter = 2000 # pose optimization step

    cam_metrics = {}
    full_metrics = {
        "PSNR": [],
        "SSIM": [],
        "LPIPS": []
    }
    #! calculate metrics per camera
    for cam, idx_list in cam_idx_list.items():
        #* Use the first frame (timestamp = 0) as the reference frame for pose optimization
        gt_image, viewpoint, _ = views[idx_list[0]]
        gt_image = gt_image.cuda()
        viewpoint = viewpoint.cuda()
        viewpoint.pseudo_timestamp = 0.0
        differentiable_cam = differentiable_cams.get_camera(viewpoint.camera_id)

        #! Stage 1: Optimize pose (2 stages, coarse to fine)
        #* Stage 1.1: Coarse Optimization
        differentiable_cam.requires_grad_(True)
        translation_lr_init = 0.001
        rotation_lr_init = 0.01
        pose_optimizer = torch.optim.Adam(
            [
                {"params": [differentiable_cam.T], "lr": translation_lr_init, 'name':'trans'},
                {"params": [differentiable_cam.q], "lr": rotation_lr_init, 'name':'rot'},
            ]
        )
        progress_bar = tqdm(range(num_iter), desc=f"Tracking Time Step")#, disable=True)
        # Keep track of best pose candidate
        candidate_q = differentiable_cam.q.clone().detach()
        candidate_T = differentiable_cam.T.clone().detach()
        current_min_loss = float(1e20)
        stop = False
        step = 0
        while True:
            if step > num_iter:
                stop = True
            rendering = differentiable_render_with_hexplane(viewpoint, differentiable_cam, gaussians, pipeline, background, apply_distortion=False)["render"]
            loss = torch.abs(gt_image - rendering).mean()
            loss.backward()
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
            if step > 10 and (loss.item() < 0.05 or torch.abs(loss - current_min_loss) < 1e-5):
                stop = True
            if loss < current_min_loss:
                current_min_loss = loss
                candidate_q = differentiable_cam.q.clone().detach()
                candidate_T = differentiable_cam.T.clone().detach()
            progress_bar.set_postfix({'loss': f"{loss.item()}"})
            progress_bar.update(1)
            step += 1
            if stop:
                break
        progress_bar.close()
        differentiable_cam.q = candidate_q.float().cuda()
        differentiable_cam.T = candidate_T.float().cuda()

        #* Stage 1.2: Fine Optimization ( Similar as the pose optimization in training stage )
        differentiable_cam.requires_grad_(True)
        translation_lr_init = opt.translation_lr_init
        translation_lr_final = opt.translation_lr_final
        rotation_lr_init = opt.rotation_lr_init
        rotation_lr_final = opt.rotation_lr_final
        pose_optimizer = torch.optim.Adam(
            [
                {"params": [differentiable_cam.T], "lr": translation_lr_init, 'name':'trans'},
                {"params": [differentiable_cam.q], "lr": rotation_lr_init, 'name':'rot'},
            ]
        )
        cam_translation_scheduler_args = get_expon_lr_func(lr_init=translation_lr_init,
                                                        lr_final=translation_lr_final,
                                                        max_steps=num_iter)
        cam_rotation_scheduler_args = get_expon_lr_func(lr_init=rotation_lr_init,
                                                            lr_final=rotation_lr_final,
                                                            max_steps=num_iter)
        def update_learning_rate(iteration):
            for param_group in pose_optimizer.param_groups:
                if param_group["name"] == "rot":
                    lr = cam_rotation_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "trans":
                    lr = cam_translation_scheduler_args(iteration)
                    param_group['lr'] = lr
        progress_bar = tqdm(range(num_iter), desc=f"Tracking Time Step")#, disable=True)
        # Keep track of best pose candidate
        candidate_q = differentiable_cam.q.clone().detach()
        candidate_T = differentiable_cam.T.clone().detach()
        current_min_loss = float(1e20)
        stop = False
        step = 0
        while True:
            if step > num_iter:
                stop = True
            update_learning_rate(step)
            rendering = differentiable_render_with_hexplane(viewpoint, differentiable_cam, gaussians, pipeline, background, apply_distortion=False)["render"]
            loss = torch.abs(gt_image - rendering).mean()
            loss.backward()
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
            if torch.abs(loss - current_min_loss) < 1e-7 and step > 100:
                stop = True
            if loss < current_min_loss:
                current_min_loss = loss
                candidate_q = differentiable_cam.q.clone().detach()
                candidate_T = differentiable_cam.T.clone().detach()
            progress_bar.set_postfix({'loss': f"{loss.item()}"})
            progress_bar.update(1)
            step += 1
            if stop:
                break
        progress_bar.close()
        differentiable_cam.q = candidate_q
        differentiable_cam.T = candidate_T

        #! Stage 2: Walk throught the data_id_list to calculate Metrics with optimized pose for current camera
        psnrs = []
        ssims = []
        lpipses = []
        renders = []
        for view_idx in tqdm(idx_list):
            batch_data = views[view_idx]
            gt_image, viewpoint, _ = batch_data
            gt_image = gt_image.cuda()
            viewpoint = viewpoint.cuda()
            viewpoint.pseudo_timestamp = 0.0
            render_pkg = differentiable_render_with_hexplane(viewpoint, differentiable_cam, gaussians, pipeline, background, apply_distortion=False)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            psnr_ = psnr(image, gt_image).mean().float().item()
            ssim_ = ssim(image, gt_image).mean().float().item()
            lpips_ = perceptual_loss(image[None].float(), gt_image[None].float()).mean().float().item()
            psnrs.append(psnr_)
            ssims.append(ssim_)
            lpipses.append(lpips_)
            if viewpoint.timestamp == 0 or cam == 'cam00':
                torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name.split(".")[0]+f"_{psnr_:.2f}.png"))
                torchvision.utils.save_image(gt_image, os.path.join(gts_path, viewpoint.image_name.split(".")[0]+".png"))
            renders.append(image[None].detach().cpu())
        renders = torch.cat(renders).permute(0, 2, 3, 1)
        video_path = render_path.replace('renders_opt', 'videos')
        os.makedirs(video_path, exist_ok=True)
        save_video(renders, os.path.join(video_path, f"{cam}.mp4"))

        psnr_ = torch.tensor(psnrs).mean().float().item()
        ssim_ = torch.tensor(ssims).mean().float().item()
        lpips_ = torch.tensor(lpipses).mean().float().item()

        logger.info(f"{cam} Mean PSNR: {psnr_:.3f}, SSIM: {ssim_:.3f}, LPIPS: {lpips_:.3f}")
        cam_metrics[cam] = {
            "SSIM": ssim_,
            "PSNR": psnr_,
            "LPIPS": lpips_,
        }
        full_metrics["SSIM"].extend(ssims)
        full_metrics["PSNR"].extend(psnrs)
        full_metrics["LPIPS"].extend(lpipses)
    for key in full_metrics.keys():
        full_metrics[key] = torch.tensor(full_metrics[key]).mean().float().item()

    with open(os.path.dirname(render_path) + "/results.json", 'w') as fp:
        json.dump(full_metrics, fp, indent=True)
    with open(os.path.dirname(render_path) + "/per_cam.json", 'w') as fp:
        json.dump(cam_metrics, fp, indent=True)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--config", type=str)

    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--target_ts_train", nargs="+", type=int, default=[-1], help="target timestamp for training views, -1 means loading all timestamp")
    parser.add_argument("--target_ts_test", nargs="+", type=int, default=[-1], help="target timestamp for testing views, -1 means loading all timestamp")
    parser.add_argument("--target_cam", nargs="+", type=str, default=[])

    args = parser.parse_args(sys.argv[1:])
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if key == "kplanes_config":
            setattr(args, key, host[key])
        elif isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(lp.extract(args), pp.extract(args), args.iteration, args.skip_train, args.skip_test,
                args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d,
                args.target_cam, args.target_ts_train, args.target_ts_test, hyper=hp.extract(args), opt=op.extract(args))
