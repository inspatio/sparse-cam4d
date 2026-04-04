from pose_utils import *
import json
import numpy as np
from argparse import ArgumentParser

def parse_transforms_mast3r(transforms):
    """
    Parse transforms from the input JSON format.
    """
    Rs = {}
    ts = {}
    image_paths = {}
    for frame in transforms:
        cam_name = frame['file_path'].split('/')[-1].split('.')[0]
        transform_matrix = np.array(frame['transform_matrix'])
        if not 'cx' in frame:
            transform_matrix[:3, 1:3] *= -1
            image_paths[cam_name] = frame['file_path']
        R = transform_matrix[:3, :3]
        t = transform_matrix[:3, 3]
        Rs[cam_name] = R
        ts[cam_name] = t
    print(f"Parsed {len(Rs)} camera poses from transforms, with cam names {sorted(list(Rs.keys()))}.")
    return Rs, ts, image_paths

def parse_transforms(transforms, opengl=False):
    """
    Parse transforms from the input JSON format.
    """
    Rs = {}
    ts = {}
    image_paths = {}
    for frame in transforms:
        fn = frame['file_path'].split('/')[-1].split('.')[0]
        cam_name, timestamp = fn.split('_')
        if timestamp != '0000':
            # continue
            if not cam_name == 'frame':
                continue
            else:
                cam_name = fn
        transform_matrix = np.array(frame['transform_matrix'])
        if opengl:
            transform_matrix[:3, 1:3] *= -1
            image_paths[cam_name] = frame['file_path']+".png"
        R = transform_matrix[:3, :3]
        t = transform_matrix[:3, 3]
        Rs[cam_name] = R
        ts[cam_name] = t
    print(f"Parsed {len(Rs)} camera poses from transforms, with cam names {sorted(list(Rs.keys()))}.")
    print(list(Rs.keys())[0], Rs[list(Rs.keys())[0]])
    return Rs, ts, image_paths

def update_sfm_transforms(transforms, Rs, ts, cams, dataset_type):
    """
    Update the transforms with new Rs and ts.
    """
    for frame in transforms:
        fn = frame['file_path'].split('/')[-1].split('.')[0]
        cam_name, timestamp = fn.split('_')
        id = cams.index(cam_name) if cam_name in cams else -1
        assert id != -1, f"Camera {cam_name} not found in cams list."
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = Rs[id]
        transform_matrix[:3, 3] = ts[id]
        frame['transform_matrix'] = transform_matrix.tolist()
        frame['file_path'] = frame['file_path']+".png"
        if dataset_type == "n3v":
            frame['time'] = frame['time'] / 10.0
        elif dataset_type == "technicolor":
            frame['time'] = int(timestamp) / 50.0
        elif dataset_type == "nvidia":
            frame['time'] = int(timestamp) / 100.0
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return transforms

if __name__ == '__main__':
    ''' Example:
    dataset_type = "n3v"
    scene = "coffee_martini"
    anchor_cam = ["cam01", "cam05", "cam10"]    #  => 计算变换的锚点相机
    sfm_train_transforms_path = f"/mnt/afs/panweihong/data/n3v/normal/{scene}/transforms_train.json" # => 待转换坐标系
    sfm_test_transforms_path = f"/mnt/afs/panweihong/data/n3v/normal/{scene}/transforms_test.json"

    dataset_type = "technicolor"
    scene = "Train"
    anchor_cam = ["cam00", "cam09", "cam15"]    #  => 计算变换的锚点相机
    sfm_train_transforms_path = f"/mnt/afs/panweihong/data/technicolor/sparse/{scene}/transforms_train.json" # => 待转换坐标系
    sfm_test_transforms_path = f"/mnt/afs/panweihong/data/technicolor/sparse/{scene}/transforms_test.json"

    dataset_type = "nvidia"
    scene = "balloon1"
    anchor_cam = ["cam06", "cam10", "cam01"]    #  => 计算变换的锚点相机
    sfm_train_transforms_path = f"/mnt/afs/panweihong/data/NvidiaDynamicScenes/normal/{scene}/transforms_train.json" # => 待转换坐标系
    sfm_test_transforms_path = f"/mnt/afs/panweihong/data/NvidiaDynamicScenes/normal/{scene}/transforms_test.json"
    '''
    parser = ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="nvidia")
    parser.add_argument("--scene", type=str, default="balloon1")
    parser.add_argument("--anchor_cam", type=str, default="cam01,cam06,cam10")
    parser.add_argument("--data_root", type=str, default="/mnt/afs/panweihong/data/NvidiaDynamicScenes/normal")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    dataset_type = args.dataset_type
    scene = args.scene
    anchor_cam = args.anchor_cam.split(",")
    sfm_train_transforms_path = f"{args.data_root}/{scene}/transforms_train.json" # => 待转换坐标系
    sfm_test_transforms_path = f"{args.data_root}/{scene}/transforms_test.json"

    #* Load sfm pose
    sfm_train_transforms = json.load(open(sfm_train_transforms_path, 'r'))
    sfm_test_transforms = json.load(open(sfm_test_transforms_path, 'r'))
    sfm_transforms_frames = sfm_test_transforms['frames'] + sfm_train_transforms['frames']
    sfm_Rs_all, sfm_ts_all, image_paths = parse_transforms(sfm_transforms_frames, opengl=True)
    sfm_Rs_anchor = np.array([sfm_Rs_all[cam] for cam in anchor_cam])
    sfm_ts_anchor = np.array([sfm_ts_all[cam] for cam in anchor_cam])
    sfm_Rs = np.array([sfm_Rs_all[cam] for cam in sfm_Rs_all])  #  => 应用变换的待转换相机
    sfm_ts = np.array([sfm_ts_all[cam] for cam in sfm_ts_all])

    #* Load optimized pose from training stage
    model_path = "/mnt/afs/afs/panweihong/panweihong/outputs/sparse4d_release_test/nvidia/balloon1"
    our_transforms_path = f"{model_path}/pose/pose_7000_final.json" # => 目标坐标系
    our_transforms = json.load(open(our_transforms_path, 'r'))
    our_Rs, our_ts, _ = parse_transforms(our_transforms)
    our_Rs_anchor = np.array([our_Rs[cam] for cam in anchor_cam])
    our_ts_anchor = np.array([our_ts[cam] for cam in anchor_cam])

    #* Pose Transformation
    # 从中心点对 (sfm_t, our_t) 估计坐标系变换 坐标系A: sfm, 坐标系B: ours 变换A→B
    def normalize_trajectory(X):
        X_mean = X.mean(0)
        X_centered = X - X_mean
        X_scale = np.linalg.norm(X_centered)
        return X_centered / X_scale, X_mean, X_scale
    sfm_ts_anchor_norm, sfm_mean, sfm_scale = normalize_trajectory(sfm_ts_anchor)
    our_ts_anchor_norm, our_mean, our_scale = normalize_trajectory(our_ts_anchor)
    scale, R_BA, t_BA = umeyama_alignment(sfm_ts_anchor_norm, our_ts_anchor_norm, sfm_Rs_anchor, our_Rs_anchor, with_scaling=True)
    assert np.linalg.det(R_BA) > 0, "The rotation matrix R_BA should be a valid rotation matrix with positive determinant."
    # 执行变换，得到在目标坐标系下的位姿
    ours_Rs_trans, ours_ts_trans = batch_transform_poses(sfm_Rs, sfm_ts, scale, R_BA, t_BA, sfm_mean, sfm_scale, our_mean, our_scale)

    #* Save
    sfm_transforms_frames_to_ours = update_sfm_transforms(sfm_transforms_frames, ours_Rs_trans, ours_ts_trans, list(sfm_Rs_all.keys()), dataset_type)
    sfm_test_transforms['frames'] = sfm_transforms_frames_to_ours
    if 'fl_x' in sfm_transforms_frames:
        sfm_test_transforms['fl_x'] = our_transforms[0]['fx']
        sfm_test_transforms['fl_y'] = our_transforms[0]['fy']
        sfm_test_transforms['cx'] = our_transforms[0]['cx']
        sfm_test_transforms['cy'] = our_transforms[0]['cy']
        sfm_test_transforms['w'] = our_transforms[0]['width']
        sfm_test_transforms['h'] = our_transforms[0]['height']
    else:
        # technicolor
        for frame in sfm_test_transforms['frames']:
            frame['fl_x'] = our_transforms[0]['fx']
            frame['fl_y'] = our_transforms[0]['fy']
            frame['width'] = our_transforms[0]['width']
            frame['height'] = our_transforms[0]['height']
            frame['cx'] = our_transforms[0]['cx']
            frame['cy'] = our_transforms[0]['cy']
    sfm_test_transforms['train_cam'] = anchor_cam
    all_cams = list(sfm_Rs_all.keys())
    sfm_test_transforms['test_cam'] = [cam for cam in all_cams if cam not in anchor_cam]

    sfm_transforms_to_ours_path = f"{model_path}/pose/sfm_transforms_to_ours.json"
    json.dump(sfm_test_transforms, open(sfm_transforms_to_ours_path, 'w'), indent=4)
    print(f"Saved {len(sfm_test_transforms['train_cam'])} train cameras and {len(sfm_test_transforms['test_cam'])} test cameras to {sfm_transforms_to_ours_path}")
