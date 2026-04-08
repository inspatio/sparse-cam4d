"""Stage 3: SfM reconstruction using COLMAP."""
import os
import sys
import shutil
import sqlite3
import struct
import subprocess
import shlex
import numpy as np
from glob import glob
from loguru import logger
from plyfile import PlyData, PlyElement


# ── COLMAP path resolution ────────────────────────────────────────────────────

def _get_colmap_cmd() -> str:
    os.environ["GLOG_minloglevel"] = "1"
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        colmap_path = os.path.join(conda_prefix, "bin", "colmap")
        if os.path.isfile(colmap_path):
            os.environ["LD_LIBRARY_PATH"] = os.path.join(conda_prefix, "lib")
            return colmap_path
        logger.warning(f"COLMAP not found at {colmap_path}, using system colmap")
    else:
        logger.warning("CONDA_PREFIX not set, using system colmap")
    return "colmap"


def _do_system(cmd_list: list) -> None:
    logger.debug("Running: " + shlex.join(cmd_list))
    result = subprocess.run(cmd_list, stdout=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (rc={result.returncode}): {shlex.join(cmd_list)}")


# ── PLY helpers ───────────────────────────────────────────────────────────────

def store_ply_with_time(path: str, xyz: np.ndarray, rgb: np.ndarray, time: np.ndarray = None) -> None:
    """Save point cloud to PLY, optionally with per-point time attribute."""
    if time is None:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                 ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                 ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        normals = np.zeros_like(xyz)
        attributes = np.concatenate((xyz, normals, rgb), axis=1)
    else:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                 ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                 ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                 ("time", "f4")]
        normals = np.zeros_like(xyz)
        attributes = np.concatenate((xyz, normals, rgb, time.reshape(-1, 1)), axis=1)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def load_ply(path: str):
    """Load PLY point cloud. Returns (positions, colors, normals, timestamps)."""
    plydata = PlyData.read(path)
    v = plydata["vertex"]
    positions = np.vstack([v["x"], v["y"], v["z"]]).T
    colors = np.vstack([v["red"], v["green"], v["blue"]]).T
    normals = np.vstack([v["nx"], v["ny"], v["nz"]]).T if "nx" in v else None
    timestamps = v["time"][:, None] if "time" in v else None
    return positions, colors, normals, timestamps


# ── COLMAP text readers ───────────────────────────────────────────────────────

def read_points3d_text(path: str):
    xyzs = rgbs = errors = None
    with open(path) as fid:
        for line in fid:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            elems = line.split()
            xyz = np.array(tuple(map(float, elems[1:4])))
            rgb = np.array(tuple(map(int, elems[4:7])))
            error = np.array(float(elems[7]))
            if xyzs is None:
                xyzs, rgbs, errors = xyz[None], rgb[None], error[None]
            else:
                xyzs = np.append(xyzs, xyz[None], axis=0)
                rgbs = np.append(rgbs, rgb[None], axis=0)
                errors = np.append(errors, error[None], axis=0)
    return xyzs, rgbs, errors


def read_points3d_binary(path: str):
    def _read(fid, n, fmt):
        return struct.unpack("<" + fmt, fid.read(n))

    xyzs = rgbs = errors = None
    with open(path, "rb") as fid:
        num_points = _read(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = _read(fid, 43, "QdddBBBd")
            xyz = np.array(props[1:4])
            rgb = np.array(props[4:7])
            error = np.array(props[7])
            if xyzs is None:
                xyzs, rgbs, errors = xyz[None], rgb[None], error[None]
            else:
                xyzs = np.append(xyzs, xyz[None], axis=0)
                rgbs = np.append(rgbs, rgb[None], axis=0)
                errors = np.append(errors, error[None], axis=0)
    return xyzs, rgbs, errors


# ── COLMAP database helper ────────────────────────────────────────────────────

_MAX_IMAGE_ID = 2 ** 31 - 1

_CREATE_ALL = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
    params BLOB, prior_focal_length INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL,
    prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
    prior_tx REAL, prior_ty REAL, prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {max_id}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
    cols INTEGER NOT NULL, data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
    cols INTEGER NOT NULL, data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
    cols INTEGER NOT NULL, data BLOB);
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
    cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL,
    F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB);
CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);
""".format(max_id=_MAX_IMAGE_ID)

_CAM_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3,
    "OPENCV": 4, "FULL_OPENCV": 5,
}


class _COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(path):
        return sqlite3.connect(path, factory=_COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_tables = lambda: self.executescript(_CREATE_ALL)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, params.tobytes(), camera_id),
        )


# ── Main SfM function ─────────────────────────────────────────────────────────

def sfm_from_scratch(
    workspace: str,
    src_image_path,
    sparse_folder: str = "sparse",
    shared_intrinsics: bool = False,
    skip: int = 1,
) -> tuple:
    """Run COLMAP SfM from scratch on images in src_image_path.

    Args:
        workspace: directory where SfM output will be written (sparse_folder/0/).
        src_image_path: path to directory of images OR list of image paths.
        sparse_folder: output subdir name under workspace (result at sparse_folder/0/).
        shared_intrinsics: if True, use single camera for all images.
        skip: use every `skip`-th image.

    Returns:
        (xyz, rgb): numpy arrays of 3D point positions and colors.
    """
    COLMAP = _get_colmap_cmd()
    sparse_dir = os.path.join(workspace, sparse_folder)
    os.makedirs(sparse_dir, exist_ok=True)

    # ── Step 1: Link images into colmap workspace ─────────────────────────────
    colmap_ws = os.path.join(workspace, "colmap_kf")
    if os.path.exists(colmap_ws):
        shutil.rmtree(colmap_ws)
    image_path = os.path.join(colmap_ws, "images")
    os.makedirs(image_path, exist_ok=True)

    if isinstance(src_image_path, list):
        for fpath in src_image_path[::skip]:
            link = os.path.join(image_path, os.path.basename(fpath))
            if not os.path.exists(link):
                os.symlink(os.path.abspath(fpath), link)
    elif os.path.isdir(src_image_path):
        files = sorted(os.listdir(src_image_path))[::skip]
        for fname in files:
            link = os.path.join(image_path, fname)
            if not os.path.exists(link):
                os.symlink(os.path.abspath(os.path.join(src_image_path, fname)), link)
    else:
        raise ValueError(f"src_image_path must be a directory or list, got: {src_image_path}")

    # ── Step 2: Feature extraction & matching ─────────────────────────────────
    db_path = os.path.join(colmap_ws, "database.db")
    single_cam = "1" if shared_intrinsics else "0"
    _do_system([COLMAP, "feature_extractor",
                "--database_path", db_path, "--image_path", image_path,
                "--ImageReader.single_camera", single_cam,
                "--SiftExtraction.use_gpu", "0"])
    _do_system([COLMAP, "exhaustive_matcher",
                "--database_path", db_path, "--SiftMatching.use_gpu", "0"])

    # ── Step 3: Sparse reconstruction ─────────────────────────────────────────
    sparse_path = os.path.join(colmap_ws, "sparse")
    sparse_0_path = os.path.join(sparse_path, "0")
    os.makedirs(sparse_path, exist_ok=True)
    _do_system([COLMAP, "mapper",
                "--database_path", db_path, "--image_path", image_path, "--output_path", sparse_path])
    _do_system([COLMAP, "model_converter",
                "--input_path", sparse_0_path, "--output_path", sparse_0_path,
                "--output_type", "TXT"])
    _do_system([COLMAP, "point_filtering",
                "--input_path", sparse_0_path, "--output_path", sparse_0_path,
                "--max_reproj_error", "2.0", "--min_track_len", "3"])

    # ── Step 4: Convert points3D.txt → points3D.ply ──────────────────────────
    txt_path = os.path.join(sparse_0_path, "points3D.txt")
    ply_path = os.path.join(sparse_0_path, "points3D.ply")
    xyz, rgb, _ = read_points3d_text(txt_path)
    if not os.path.exists(ply_path):
        store_ply_with_time(ply_path, xyz, rgb)

    # ── Step 5: Copy results to output sparse_folder ─────────────────────────
    _do_system(["cp", "-r", sparse_0_path, sparse_dir])
    _do_system(["cp", db_path, os.path.join(sparse_dir, "0")])

    # ── Step 6: Check registration completeness ───────────────────────────────
    n_input = len(os.listdir(image_path))
    images_txt = os.path.join(sparse_0_path, "images.txt")
    n_registered = sum(
        1 for line in open(images_txt)
        if line.strip() and not line.startswith("#") and len(line.split()) >= 9
    ) // 2
    if n_registered != n_input:
        logger.warning(
            f"SfM: only {n_registered}/{n_input} images registered. "
            "Transforms for unregistered images will be missing."
        )

    shutil.rmtree(colmap_ws)
    logger.info(f"SfM done. {len(xyz)} points, {n_registered}/{n_input} images registered → {sparse_dir}/0/")
    return xyz, rgb


def run_sfm(
    workspace: str,
    diffusion_dir: str = "diffusion",
    sparse_folder: str = "sparse",
    shared_intrinsics: bool = False,
) -> tuple:
    """Convenience wrapper: run SfM on workspace/diffusion_dir images."""
    src_image_path = os.path.join(workspace, diffusion_dir)
    assert os.path.isdir(src_image_path), f"Image dir not found: {src_image_path}"
    return sfm_from_scratch(
        workspace, src_image_path,
        sparse_folder=sparse_folder,
        shared_intrinsics=shared_intrinsics,
    )
