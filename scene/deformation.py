import torch
import torch.nn as nn
import torch.nn.init as init
from utils.graphics_utils import batch_quaternion_multiply, axis_angle_to_quaternion
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
from loguru import logger
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.args = args
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        #* FiLM modulate
        self.apply_film_modulate = args.apply_film_modulate

        self.ratio=0
        self.create_net()

    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        # print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def set_max_delta(self, max_dx, max_ds, max_dr=0.1):
        self.max_dx = max_dx
        self.max_ds = max_ds
        self.max_dr = max_dr

    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:

            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]

        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)

        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        if self.args.apply_rotation:
            self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        else:
            self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

        self.activation = nn.Sequential(nn.Tanh())

        if self.apply_film_modulate:
            #* FiLM modulate
            # FiLM conditioning layer from (t1, t2) -> gamma, beta (1xW each)
            self.film_layer = nn.Sequential(nn.Linear(2, self.W * 2),) # output is [gamma, beta]
            # shared FiLM modulation
            self.modulation = lambda h, gamma, beta: h * gamma + beta

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):
        #! normalize timestamps, [0,1) -> [-1,1)
        time_emb = time_emb * 2 - 1

        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            # grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb)
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1)

        hidden = self.feature_out(hidden)

        if self.apply_film_modulate:
            #* FiLM modulate
            gamma_beta = self.film_layer(time_emb)  # [B, 2W]
            gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, W] each
            h_modulated = self.modulation(hidden, gamma, beta)
            return h_modulated

        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        dx, ds, dr, do, dshs = None, None, None, None, None
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)

        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)

        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden) # direct
            if torch.isnan(dx).any():
                logger.warning(f"{torch.isnan(dx).any(dim=1).sum()} dx has nan")
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx

        if self.args.no_ds :
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            if torch.isnan(ds).any():
                logger.warning(f"{torch.isnan(ds).any(dim=1).sum()} ds has nan")
            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds

        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            if torch.isnan(dr).any():
                logger.warning(f"{torch.isnan(dr).any(dim=1).sum()} dr has nan")
            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                d_quat = axis_angle_to_quaternion(dr) # mlp output axis-angle instead of delta_quat
                rotations = batch_quaternion_multiply(rotations_emb[:,:4]*mask, d_quat)
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:1]
        else:
            do = self.opacity_deform(hidden)
            if torch.isnan(do).any():
                logger.warning(f"{torch.isnan(do).any(dim=1).sum()} do has nan")
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do

        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])
            if torch.isnan(dshs).any():
                logger.warning(f"{torch.isnan(dshs).any(dim=1).sum()} dshs has nan")
            shs = torch.zeros_like(shs_emb)
            shs = shs_emb*mask.unsqueeze(-1) + dshs

        return pts, scales, rotations, opacity, shs, dx, ds, dr, do, dshs
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list

class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe # always 0
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(nn.Linear(times_ch, timenet_width), nn.ReLU(),nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))

        self.apply(initialize_weights)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio

    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        means3D, scales, rotations, opacity, shs, dx, ds, dr, do, dshs = self.deformation_net( point_emb,
                                                scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel)
        return means3D, scales, rotations, opacity, shs, dx, ds, dr, do, dshs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()
    def toggle_parameters(self, requires_grad=True):
        for param in self.get_mlp_parameters() + self.get_grid_parameters():
            param.requires_grad = requires_grad

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
def poc_fre(input_data,poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb