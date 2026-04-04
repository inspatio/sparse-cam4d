import itertools
from typing import Optional, Sequence, Iterable, Collection

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:])) # B, 1, N, 2
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    #! F.grid_sampler 坐标映射逻辑
    #* grid = [B, feature_dim, *res] = [1, 16, 301, 50] & align_corners == True
    #* H = 301: -1 -> 0, 1 -> 300; W = 50: -1 -> 0, 1 -> 49
    #* 坐标映射公式：index = (normalized_coord + 1) * (size - 1) / 2
    #* ts=150, t=150/300=0.5, normalized_t=0.5*2-1=0, index=(0+1)*(301-1)/2=150 ✅
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]

    if torch.isnan(interp).any():
        logger.warning(f"interp feature {interp.shape} has {torch.isnan(interp).any(dim=1).sum()} nan")
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    # has_time_planes = in_dim == 4
    has_time_planes = in_dim >= 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    if in_dim == 5:
        coo_combs = coo_combs[:-1]
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and (3 in coo_comb or 4 in coo_comb):  # Initialize time planes to 1，time planes 初始化为 1 为了放大动态维度的响应
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        # nn.init.uniform_(new_grid_coef, a=-0.01, b=0.01)

        grid_coefs.append(new_grid_coef)

    if in_dim == 5:
        assert len(grid_coefs) == 9

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci in range(len(grid)):
            coo_comb = coo_combs[ci]
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp

# def interpolate_ms_features_batchify(
#     pts: torch.Tensor,
#     ms_grids: Collection[Iterable[nn.Module]],
#     grid_dimensions: int,
#     concat_features: bool,
#     num_levels: Optional[int],
# ) -> torch.Tensor:
#     coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
#     if num_levels is None:
#         num_levels = len(ms_grids)
#     multi_scale_interp = [] if concat_features else 0.
#     # grid_dimensions = 5 ===> (x,y,z,t,v) ===> 9-planes (xy xz xt xv yz yt yv zt zv)
#     for scale_id,  grid in enumerate(ms_grids[:num_levels]):
#         interp_space = 1.
#         # -------- spatial grid (xy, xz, yz) or (01, 02, 12): indices = [0, 1, 4] -------- #
#         spatial_idx = [i for i, cc in enumerate(coo_combs) if not (3 in cc or 4 in cc)]
#         # -------- temporal grid (xt, yt, zt) or (03, 13, 23): indices = [2, 5, 7] -------- #
#         temporal_idx = [i for i, cc in enumerate(coo_combs) if 3 in cc and not 4 in cc]
#         # -------- view grid (xv, yv, zv) or (04, 14, 24): indices = [3, 6, 8] -------- #
#         view_idx = [i for i, cc in enumerate(coo_combs) if 4 in cc and not 3 in cc]

#         feature_dim = grid[0].shape[1]
#         N = pts.shape[0]
#         def batch_sample(indices):
#             # grids: list of (1, C, H, W) with same H, W within group
#             grids_cat = torch.cat([grid[i] for i in indices], dim=0) # (K, C, H, W)
#             coords_cat = torch.stack([pts[..., coo_combs[i]] for i in indices], dim=0).unsqueeze(1) # (K, 1, N, 2)
#             out = F.grid_sample(grids_cat, coords_cat, align_corners=True, mode='bilinear', padding_mode='border') # (K, C, 1, N)
#             return out.view(len(indices), feature_dim, N).permute(2, 0, 1) # (N, K, C)

#         groups = [spatial_idx, temporal_idx] + ([view_idx] if view_idx else [])
#         outs = [batch_sample(idx) for idx in groups]  # each: (N, K, C)

#         # product over all planes: (N, C)
#         interp_space = outs[0].prod(dim=1)
#         for o in outs[1:]:
#             interp_space = interp_space * o.prod(dim=1)

#         # combine over scales
#         if concat_features:
#             multi_scale_interp.append(interp_space)
#         else:
#             multi_scale_interp = multi_scale_interp + interp_space

#     if concat_features:
#         multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
#     return multi_scale_interp

def interpolate_ms_features_batchify(
    pts: torch.Tensor,
    ms_grids: Collection[Iterable[nn.Module]],
    concat_features: bool,
    num_levels: Optional[int],
    coo_combs: list,
    plane_groups: list,
) -> torch.Tensor:
    """
    coo_combs   : precomputed list(itertools.combinations(range(in_dim), grid_dimensions))
    plane_groups: precomputed list of index-lists, e.g. [spatial_idx, temporal_idx] or
                  [spatial_idx, temporal_idx, view_idx].  Passed in so that torch.compile
                  never sees range(pts.shape[-1]) with a symbolic size.
    """
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.

    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        feature_dim = grid[0].shape[1]
        N = pts.shape[0]

        def batch_sample(indices):
            grids_cat = torch.cat([grid[i] for i in indices], dim=0)                    # (K, C, H, W)
            coords_cat = torch.stack([pts[..., coo_combs[i]] for i in indices], dim=0).unsqueeze(1)  # (K, 1, N, 2)
            out = F.grid_sample(grids_cat, coords_cat, align_corners=True,
                                mode='bilinear', padding_mode='border')                  # (K, C, 1, N)
            return out.view(len(indices), feature_dim, N).permute(2, 0, 1)              # (N, K, C)

        outs = [batch_sample(idx) for idx in plane_groups]

        # product over all planes: (N, C)
        interp_space = outs[0].prod(dim=1)
        for o in outs[1:]:
            interp_space = interp_space * o.prod(dim=1)

        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp

class HexPlaneField(nn.Module):
    def __init__(
        self,
        bounds,
        planeconfig,
        multires,
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]

            self.grids.append(gp)

        # 2. Precompute index structures for interpolate_ms_features_batchify
        in_dim = planeconfig["input_coordinate_dim"]
        grid_nd = planeconfig["grid_dimensions"]
        self._coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
        if in_dim == 5:
            self._coo_combs = self._coo_combs[:-1]  # mirror init_grid_param exclusion
        spatial_idx  = [i for i, cc in enumerate(self._coo_combs) if not (3 in cc or 4 in cc)]
        temporal_idx = [i for i, cc in enumerate(self._coo_combs) if 3 in cc and not 4 in cc]
        view_idx     = [i for i, cc in enumerate(self._coo_combs) if 4 in cc and not 3 in cc]
        self._plane_groups = [spatial_idx, temporal_idx] + ([view_idx] if view_idx else [])

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min, margin_ratio = 0.1):
        center = (xyz_min + xyz_max) / 2
        half_range = (xyz_max - xyz_min) / 2
        margin = half_range * margin_ratio
        new_half_range = half_range + margin
        aabb = torch.tensor([
            center + new_half_range,
            center - new_half_range
        ],dtype=torch.float32).cuda()

        self.aabb = nn.Parameter(aabb,requires_grad=False)
        # print(f"Voxel Plane: set aabb={self.aabb}, margin={margin}")

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities.
        timestamps: [-1,1]
        """
        pts = normalize_aabb(pts, self.aabb) # pts -> [-1,1]

        #! safety check: check if the coordinates are out of bounds
        #* 检查坐标越界[-1,1] *#
        over = (torch.abs(pts) > 1.0).any(dim=1).sum().item()
        total = pts.shape[0]
        if over / total > 0.1:
            logger.warning(f"invalid points {over}/{total}, min_pts={pts.min().item()} max_pts={pts.max().item()}" )
        if torch.isnan(pts).any():
            logger.warning(f"pts {pts.shape} has {torch.isnan(pts).any(dim=1).sum()} nan, min_pts={pts.min().item()} max_pts={pts.max().item()}")
        if torch.isinf(pts).any():
            logger.warning(f"pts {pts.shape} has {torch.isinf(pts).any(dim=1).sum()} inf, min_pts={pts.min().item()} max_pts={pts.max().item()}")
        #* 检查时间坐标越界[-1,1] *#
        over = (torch.abs(timestamps) > 1.0).any(dim=1).sum().item()
        total = timestamps.shape[0]
        if over / total > 0.1:
            logger.warning(f"invalid timestamps {over}/{total}")
        if torch.isnan(timestamps).any():
            logger.warning(f"timestamps {timestamps.shape} has {torch.isnan(timestamps).any(dim=1).sum()} nan, min_pts={timestamps.min().item()} max_pts={timestamps.max().item()}")
        if torch.isinf(timestamps).any():
            logger.warning(f"timestamps {timestamps.shape} has {torch.isinf(timestamps).any(dim=1).sum()} inf, min_pts={timestamps.min().item()} max_pts={timestamps.max().item()}")
        #* 检查grid plane has nan *#
        for i, p in enumerate(self.grids[0]):
            if torch.isnan(p).any():
                logger.warning(f"grid plane #{i} has {torch.isnan(p).sum()} nan")

        pts = torch.clamp(pts, -1.0 + 1e-6, 1.0 - 1e-6)
        # timestamps = torch.clamp(timestamps, -1.0, 0.9999)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]


        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features_batchify(
            pts, ms_grids=self.grids,  # noqa
            concat_features=self.concat_features, num_levels=None,
            coo_combs=self._coo_combs, plane_groups=self._plane_groups)

        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)

        if torch.isnan(features).any():
            logger.warning(f"interpolate_ms_features {features.shape} has {torch.isnan(features).sum()} nan")

        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features


if __name__ == "__main__":
    # test interpolate_ms_features_batchify func. see if the results equal to interpolate_ms_features func
    import sys
    from time import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    def make_grids(multires, planeconfig):
        grids = nn.ModuleList()
        for res in multires:
            config = planeconfig.copy()
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            grids.append(gp)
        return grids.to(device)

    def run_test(label, planeconfig, multires, N=4000000):
        print(f"\n[{label}]  in_dim={planeconfig['input_coordinate_dim']}  multires={multires}  N={N}")
        grids = make_grids(multires, planeconfig)
        pts = torch.rand(N, planeconfig["input_coordinate_dim"], device=device) * 2 - 1  # [-1,1]

        in_dim = planeconfig["input_coordinate_dim"]
        grid_nd = planeconfig["grid_dimensions"]
        coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
        if in_dim == 5:
            coo_combs = coo_combs[:-1]
        spatial_idx  = [i for i, cc in enumerate(coo_combs) if not (3 in cc or 4 in cc)]
        temporal_idx = [i for i, cc in enumerate(coo_combs) if 3 in cc and not 4 in cc]
        view_idx     = [i for i, cc in enumerate(coo_combs) if 4 in cc and not 3 in cc]
        plane_groups = [spatial_idx, temporal_idx] + ([view_idx] if view_idx else [])

        t0 = time()
        out_ref = interpolate_ms_features(
            pts, ms_grids=grids,
            grid_dimensions=planeconfig["grid_dimensions"],
            concat_features=True, num_levels=None)
        t1 = time()
        out_new = interpolate_ms_features_batchify(
            pts, ms_grids=grids,
            concat_features=True, num_levels=None,
            coo_combs=coo_combs, plane_groups=plane_groups)
        t2 = time()

        assert out_ref.shape == out_new.shape, f"shape mismatch: {out_ref.shape} vs {out_new.shape}"
        max_err = (out_ref - out_new).abs().max().item()
        mean_err = (out_ref - out_new).abs().mean().item()
        ok = max_err < 1e-5
        print(f"  shape={out_ref.shape}  max_err={max_err:.2e}  mean_err={mean_err:.2e}  {'PASS' if ok else 'FAIL'} time={t1-t0:.4f}s -> {t2-t1:.4f}s")
        if not ok:
            print(f"  ref[:3,:4] = {out_ref[:3, :4]}")
            print(f"  new[:3,:4] = {out_new[:3, :4]}")
        return ok

    all_pass = True

    # Case 1: 4D xyzt  (standard HexPlane, in_dim=4, 6 planes)
    cfg_4d = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 16,
        'resolution': [64, 64, 64, 101],
    }
    all_pass &= run_test("4D xyzt, multires=[1,2]",   cfg_4d, [1, 2])
    all_pass &= run_test("4D xyzt, multires=[1,2]",   cfg_4d, [1, 2])
    all_pass &= run_test("4D xyzt, multires=[1]",     cfg_4d, [1])
    all_pass &= run_test("4D xyzt, multires=[1,2,4]", cfg_4d, [1, 2, 4])

    # Case 2: 5D xyzTV  (in_dim=5, 9 planes: spatial + temporal + view)
    cfg_5d = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 5,
        'output_coordinate_dim': 16,
        'resolution': [64, 64, 64, 101, 25],
    }
    all_pass &= run_test("5D xyzTV, multires=[1,2]", cfg_5d, [1, 2])

    print(f"\n{'All tests PASSED' if all_pass else 'Some tests FAILED'}")
    sys.exit(0 if all_pass else 1)
