import numpy as np
import torch
from utils.graphics_utils import getWorld2View3, getProjectionMatrix, getProjectionMatrixCenterShift
from kornia import create_meshgrid
from copy import deepcopy
from torch.nn import functional as F

class DifferentiableCamera:
    def __init__(self,
                 camera_id, R, T, width, height, cx, cy, fl_x, fl_y, data_device = "cuda",
                 zfar = 100.0, znear = 0.01, trans=np.array([0.0, 0.0, 0.0]), scale=1.0
    ):
        self.camera_id = camera_id
        self.data_device = data_device
        self.width = width
        self.height = height
        self.cx = torch.tensor(cx).float().to(self.data_device)
        self.cy = torch.tensor(cy).float().to(self.data_device)
        self.fl_x = torch.tensor(fl_x).float().to(self.data_device)
        self.fl_y = torch.tensor(fl_y).float().to(self.data_device)
        self.zfar = zfar
        self.znear = znear
        self.trans = torch.tensor(trans).to(self.data_device)
        self.scale = scale

        self.T = torch.tensor(T).float().to(self.data_device)
        self.q = self.matrix_to_quaternion(torch.tensor(R)).reshape(4, ).float().to(self.data_device)

        self.T_origin = deepcopy(self.T).detach().to(self.data_device).requires_grad_(False)
        self.q_origin = deepcopy(self.q).detach().to(self.data_device).requires_grad_(False)

    # #------------------------------ Camera Properties -----------------------------# #
    @staticmethod
    def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    @staticmethod
    def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

        def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
            ret = torch.zeros_like(x)
            positive_mask = x > 0
            ret[positive_mask] = torch.sqrt(x[positive_mask])
            return ret

        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
        )

        q_abs = _sqrt_positive_part(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            )
        )

        quat_by_rijk = torch.stack(
            [
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )

        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        return quat_candidates[
            F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
        ].reshape(batch_dim + (4,))

    @property
    def R(self):
        return self.quaternion_to_matrix(self.q).float().to(self.data_device)

    @property
    def FoVx(self):
        return 2 * torch.atan(self.width / (2 * self.fl_x))

    @property
    def FoVy(self):
        return 2 * torch.atan(self.height / (2 * self.fl_y))

    @property
    def projection_matrix(self):
        if self.cx > 0:
            return getProjectionMatrixCenterShift(self.znear, self.zfar, self.cx, self.cy, self.fl_x, self.fl_y, self.width, self.height).transpose(0,1).float().to(self.data_device)
        else:
            return getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).float().to(self.data_device)

    @property
    def world_view_transform(self):
        return getWorld2View3(self.R, self.T, self.trans, self.scale).transpose(0, 1).float().to(self.data_device)

    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]
    # #------------------------------ Camera Properties -----------------------------# #

    def requires_grad_(self, requires_grad=False):
        self.T.requires_grad_(requires_grad)
        self.q.requires_grad_(requires_grad)

    def get_rays(self):
        grid = create_meshgrid(self.height, self.width, normalized_coordinates=False, device=self.data_device)[0] + 0.5
        i, j = grid.unbind(-1)
        pts_view = torch.stack([(i-self.cx)/self.fl_x, (j-self.cy)/self.fl_y, torch.ones_like(i), torch.ones_like(i)], -1).to(self.data_device)
        c2w = torch.linalg.inv(self.world_view_transform.transpose(0, 1))
        pts_world =  pts_view @ c2w.T
        directions = pts_world[...,:3] - self.camera_center[None,None,:]
        return self.camera_center[None,None], directions / torch.norm(directions, dim=-1, keepdim=True)

import torch.nn as nn
class DifferentiableCameras(nn.Module):
    def __init__(self, cam_infos, resolution_scale, args):
        super().__init__()
        self.unique_camera_ids = [] # Camera ID to Index Mapping
        #! Initialize Camera Parameters from Any Timestamp
        self.cameras = {}
        self.is_true_image = {}
        for cam_info in cam_infos:
            cid = cam_info.camera_id
            if cid not in self.cameras:
                orig_w, orig_h = cam_info.width, cam_info.height

                if args.resolution in [1, 2, 3, 4, 8]:
                    resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
                    scale = resolution_scale * args.resolution
                else:  # should be a type that converts to float
                    if args.resolution == -1:
                        if orig_w > 1600:
                            global WARNED
                            if not WARNED:
                                print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                                    "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                                WARNED = True
                            global_down = orig_w / 1600
                        else:
                            global_down = 1
                    else:
                        global_down = orig_w / args.resolution

                    scale = float(global_down) * float(resolution_scale)
                    resolution = (int(orig_w / scale), int(orig_h / scale))

                cx = cam_info.cx / scale
                cy = cam_info.cy / scale
                fl_y = cam_info.fl_y / scale
                fl_x = cam_info.fl_x / scale

                self.unique_camera_ids.append(cid)
                self.cameras[cid] = DifferentiableCamera(
                    camera_id=cid,
                    R = cam_info.R,
                    T = cam_info.T,
                    width = resolution[0],
                    height = resolution[1],
                    cx = cx, cy = cy, fl_x = fl_x, fl_y = fl_y,
                    data_device=args.data_device
                )
                self.is_true_image[cid] = cam_info.is_true_image
    def get_camera(self, camera_id):
        if camera_id in self.cameras:
            return self.cameras[camera_id]
        else:
            return None

    def is_true_camera(self, camera_id):
        if camera_id in self.is_true_image:
            return self.is_true_image[camera_id]
        else:
            return False