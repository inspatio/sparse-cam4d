import numpy as np

def rotation_matrix_from_vectors(a, b, eps=1e-8):
    """
    Compute rotation matrix that aligns vector a to vector b (both 3D).
    Minimal rotation. Handles anti-parallel case.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cross = np.cross(a, b)
    s = np.linalg.norm(cross)
    c = np.dot(a, b)
    if s < eps:
        # vectors are parallel or anti-parallel
        if c > 0:
            return np.eye(3)
        else:
            # 180 degree rotation: choose arbitrary orthogonal axis
            # find any vector orthogonal to a
            if abs(a[0]) < 0.9:
                ortho = np.array([1.0, 0.0, 0.0])
            else:
                ortho = np.array([0.0, 1.0, 0.0])
            v = ortho - a * np.dot(a, ortho)
            v = v / np.linalg.norm(v)
            # rotation of 180 deg about axis v: R = I - 2 * v v^T
            return np.eye(3) - 2 * np.outer(v, v)
    # Rodrigues formula
    k = cross
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
    return R

from scipy.spatial.transform import Rotation as Rscipy
from scipy.optimize import least_squares
def refine_global_transform(src_pts, dst_pts, src_Rs, dst_Rs, R_init, t_init, scale_init=1.0):
    """用最小二乘优化全局 R、t、scale"""
    # # 根据归一化点位误差设权重
    # rot_errs = []
    # for Rs, Rd in zip(src_Rs, dst_Rs):
    #     R_pred = R_init @ Rs
    #     R_diff = Rd @ R_pred.T
    #     ang = np.linalg.norm(Rscipy.from_matrix(R_diff).as_rotvec())  # rad
    #     rot_errs.append(ang)
    # mean_rot_err = np.mean(rot_errs)
    # pos_err_est = np.linalg.norm(dst_pts - (scale_init * (R_init @ src_pts.T).T + t_init), axis=1).mean()
    # pose_weight = pos_err_est / (mean_rot_err + 1e-12)
    # print("Pose weight:", pose_weight)
    pose_weight = 0.05

    def residual(params):
        rotvec = params[0:3]
        t = params[3:6]
        scale = params[6]
        R = Rscipy.from_rotvec(rotvec).as_matrix()
        pred = scale * (R @ src_pts.T).T + t
        res_points = (pred - dst_pts).ravel()

        rot_errs = []
        for Rs, Rd in zip(src_Rs, dst_Rs):
            R_pred = R @ Rs
            R_diff = Rd @ R_pred.T
            rv_err = Rscipy.from_matrix(R_diff).as_rotvec()  # 3-dim
            rot_errs.append(rv_err)
        res_pose = pose_weight * np.concatenate(rot_errs, axis=0)
        return np.hstack([res_points, res_pose])

    rotvec_init = Rscipy.from_matrix(R_init).as_rotvec()
    x0 = np.hstack([rotvec_init, t_init, scale_init])
    result = least_squares(residual, x0, method='trf')

    R_opt = Rscipy.from_rotvec(result.x[0:3]).as_matrix()
    t_opt = result.x[3:6]
    scale_opt = result.x[6]
    return scale_opt, R_opt, t_opt

def umeyama_alignment(src_pts, dst_pts, src_Rs=None, dst_Rs=None, with_scaling=True):
    """
    从对应点中估计 similarity transform (scale, rotation, translation)
    """
    assert src_pts.shape == dst_pts.shape
    n = src_pts.shape[0]

    mean_src = np.mean(src_pts, axis=0)
    mean_dst = np.mean(dst_pts, axis=0)

    if n == 2:
        # direction-based rotation
        print("src pts:")
        print(src_pts)
        print("dst pts:")
        print(dst_pts)
        vA = src_pts[1] - src_pts[0]
        vB = dst_pts[1] - dst_pts[0]
        if np.linalg.norm(vA) < 1e-8 or np.linalg.norm(vB) < 1e-8:
            R = np.eye(3)
        else:
            R = rotation_matrix_from_vectors(vA, vB)
        if with_scaling:
            normA = np.linalg.norm(vA)
            normB = np.linalg.norm(vB)
            scale = normB / normA if normA > 0 else 1.0
        else:
            scale = 1.0
        t = dst_pts[0] - scale * (R @ src_pts[0])

        if src_Rs is not None:
            print("refine global transform")
            print("before refinement")
            print("Scale:", scale)
            print("R_BA:\n", R)
            print("t_BA:", t)
            scale, R, t = refine_global_transform(src_pts, dst_pts, src_Rs, dst_Rs, R, t, scale)
        return scale, R, t
    else:
        # dim = src_pts.shape[1]
        # src_centered = src_pts - mean_src
        # dst_centered = dst_pts - mean_dst
        # cov_matrix = dst_centered.T @ src_centered / n
        # U, D, Vt = np.linalg.svd(cov_matrix)
        # S = np.eye(dim)
        # if np.linalg.det(U @ Vt) < 0:
        #     S[-1, -1] = -1
        # R_mat = U @ S @ Vt
        # if with_scaling:
        #     var_src = np.var(src_centered, axis=0).sum()
        #     scale = (D @ S).sum() / var_src
        # else:
        #     scale = 1.0
        # t = mean_dst - scale * R_mat @ mean_src

        def average_rotations(rot_mats, weights=None):
            """简单的旋转平均：在旋转向量空间里做加权平均"""
            rots = Rscipy.from_matrix(rot_mats)
            rvs = rots.as_rotvec()
            if weights is None:
                rv_mean = rvs.mean(axis=0)
            else:
                w = np.asarray(weights).reshape(-1, 1)
                rv_mean = (w * rvs).sum(axis=0) / (w.sum() + 1e-12)
            return Rscipy.from_rotvec(rv_mean).as_matrix()
        scale = 1.0
        src_Rs = np.asarray(src_Rs, dtype=np.float64)
        dst_Rs = np.asarray(dst_Rs, dtype=np.float64)
        assert src_Rs.shape == dst_Rs.shape and src_Rs.shape[-2:] == (3, 3)
        # 理论上有:  dst_R ≈ R_global @ src_R
        Rg_candidates = np.einsum('nij,njk->nik', dst_Rs, np.transpose(src_Rs, (0,2,1)))
        R = average_rotations(Rg_candidates)  # 平均得到一个全局旋转初值
        t = mean_dst - scale * R @ mean_src
        if src_Rs is not None:
            print("refine global transform")
            print("before refinement")
            print("Scale:", scale)
            print("R_BA:\n", R)
            print("t_BA:", t)
            scale, R_mat, t = refine_global_transform(src_pts, dst_pts, src_Rs, dst_Rs, R, t, scale)

        return scale, R_mat, t

def transform_pose(R_A, t_A, scale, R_BA, t_BA, A_mean=None, A_scale=None, B_mean=None, B_scale=None):
    """
    对单个相机位姿进行变换：从坐标系 A → B
    """
    R_B = R_BA @ R_A
    if A_mean is not None:
        t_A_norm = (t_A - A_mean) / A_scale
        t_B_norm = scale * (R_BA @ t_A_norm) + t_BA
        t_B = t_B_norm * B_scale + B_mean
    else:
        t_B = scale * (R_BA @ t_A) + t_BA
    return R_B, t_B

def batch_transform_poses(Rs_A, ts_A, scale, R_BA, t_BA, A_mean=None, A_scale=None, B_mean=None, B_scale=None):
    """
    对多个相机进行批量坐标变换
    """
    Rs_B = []
    ts_B = []
    for R_A, t_A in zip(Rs_A, ts_A):
        R_B, t_B = transform_pose(R_A, t_A, scale, R_BA, t_BA, A_mean, A_scale, B_mean, B_scale)
        Rs_B.append(R_B)
        ts_B.append(t_B)
    return np.array(Rs_B), np.array(ts_B)