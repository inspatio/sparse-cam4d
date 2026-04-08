import torch
import numpy as np
import os
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from typing import List, Union

from utils.pvd_utils import save_render
from utils.diffusion_utils import instantiate_from_config, load_model_checkpoint, image_guided_synthesis


class ViewCrafterParams:
    def __init__(self, ckpt_path=None, config_path=None, mode='sparse', device='cuda:0'):
        self.device = device
        self.mode = 'sparse_view_interp'
        self.elevation = 5.0
        self.center_scale = 1.0
        self.d_theta = [10]
        self.d_phi = [30]
        self.d_r = [-0.2]
        self.d_x = [0.0]
        self.d_y = [0.0]
        self.mask_image = False
        self.mask_pc = True
        self.reduce_pc = False
        self.bg_trd = 0.2
        self.dpt_trd = 1.0
        self.ddim_steps = 50
        self.ddim_eta = 1.0
        self.bs = 1
        self.height = 576
        self.width = 1024
        self.frame_stride = 10
        self.unconditional_guidance_scale = 7.5
        self.seed = 123
        self.video_length = 25
        self.negative_prompt = False
        self.text_input = True
        self.prompt = 'Rotating view of a scene'
        self.multiple_cond_cfg = False
        self.cfg_img = None
        self.timestep_spacing = "uniform_trailing"
        self.guidance_rescale = 0.7
        self.perframe_ae = True
        self.n_samples = 1
        self.batch_size = 1
        self.schedule = 'linear'
        self.niter = 300
        self.lr = 0.01
        self.min_conf_thr = 3.0

        _here = os.path.dirname(os.path.abspath(__file__))
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path
        else:
            ckpt_name = 'model_sparse.ckpt' if mode == 'sparse' else 'model.ckpt'
            self.ckpt_path = os.path.join(_here, 'checkpoints', ckpt_name)

        self.config = config_path if config_path is not None else os.path.join(_here, 'configs', 'inference_pvd_1024.yaml')


class ViewCrafter:
    def __init__(self, save_dir=None, mode='sparse', ckpt_path=None, config_path=None, device='cuda:0'):
        self.opts = ViewCrafterParams(ckpt_path=ckpt_path, config_path=config_path, mode=mode, device=device)
        if save_dir and os.path.exists(save_dir):
            self.opts.save_dir = save_dir
        else:
            print("save_dir is None or not exist, call set_save_dir() to set save_dir")
        self.device = self.opts.device
        self.setup_diffusion()

    def set_save_dir(self, save_dir):
        assert os.path.exists(save_dir)
        self.opts.save_dir = save_dir

    def setup_diffusion(self):
        seed_everything(self.opts.seed)

        assert os.path.exists(self.opts.config), f"Error: config {self.opts.config} Not Found!"
        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())

        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        self.model_config = model_config
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model

        h, w = self.opts.height // 8, self.opts.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]

    def reset_diffusion(self):
        model = instantiate_from_config(self.model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model

    def run_diffusion(self, renderings, condition_index=[0]):
        prompts = [self.opts.prompt]
        videos = (renderings * 2. - 1.).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_samples = image_guided_synthesis(
                self.diffusion, prompts, videos, self.noise_shape, self.opts.n_samples,
                self.opts.ddim_steps, self.opts.ddim_eta, self.opts.unconditional_guidance_scale,
                self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input,
                self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale,
                condition_index
            )
        return torch.clamp(batch_samples[0][0].permute(1, 2, 3, 0), -1., 1.)

    def run(self, render_results: Union[List, torch.Tensor], save=False, condition_index=[0]):
        with torch.no_grad():
            if isinstance(render_results, List):
                assert len(render_results) == 25
                render_results = torch.stack(render_results, dim=0).float().cuda()
                if save:
                    save_render(render_results, os.path.join(self.opts.save_dir, 'pts_render'))
            self.shape_ori = render_results[0].shape[:2]
            render_results = F.interpolate(
                render_results.permute(0, 3, 1, 2), size=(576, 1024),
                mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)
            diffusion_results = self.run_diffusion(render_results, condition_index)
            diffusion_results = F.interpolate(
                diffusion_results.permute(0, 3, 1, 2),
                size=(self.shape_ori[0], self.shape_ori[1]),
                mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)
            diffusion_results = (diffusion_results + 1.0) / 2.0
            if save:
                save_render(diffusion_results, os.path.join(self.opts.save_dir, 'images'))
            return diffusion_results
