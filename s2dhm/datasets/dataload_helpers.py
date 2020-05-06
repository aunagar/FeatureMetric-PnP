import os
import sys
import numpy as np
import math
import pickle
from glob import glob
from typing import List
from pathlib import Path
from collections import namedtuple

sys.path.append('s2dhm/datasets/')

reconstruction_data = namedtuple('reconstruction_data',
    'intrinsics distortion_coefficients points_2D points_3D')

def key_converter(filename: str):
    """Convert an absolute filename the keys format in the 3D files."""
    return '/'.join(filename.split('/')[-3:])

def _assemble_intrinsics(focal, cx, cy, distortion):
    """Assemble intrinsics matrix from parameters."""
    intrinsics = np.eye(3)
    intrinsics[0,0] = float(focal)
    intrinsics[1,1] = float(focal)
    intrinsics[0,2] = float(cx)
    intrinsics[1,2] = float(cy)
    distortion_coefficients = np.array([float(distortion), 0.0, 0.0, 0.0])
    return intrinsics, distortion_coefficients

def quaternion_matrix(quaternion: List, eps=np.finfo(float).eps * 4.0):
    """Return homogeneous rotation matrix from quaternion."""
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < eps:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def assemble_matrix(R: np.ndarray, T: np.ndarray):
    """Assemble a 4x4 transformation matrix."""
    M = np.zeros((4, 4))
    M[ 3, 3] = 1.0
    M[:3,:3] = np.array(R)
    M[:3, 3] = np.array(T)
    return M

def from_nvm(nvm_path: str, filenames: str):
    """ Parse ground truth poses from an .nvm file.

    Args:
        nvm_path: The path to the .nvm file.
        filenames: files for which you want to load the pose
    Returns:
        filenames to pose mapping
    """
    # Parse .nvm file
    nvm_poses = [line.rstrip('\n') for line in open(nvm_path)]
    # Load poses and reference filename_to_intrinsics from the .NVM file
    fname_to_pose = dict()
    filenames = [key_converter(f) for f in filenames]
    for i in range(int(nvm_poses[2])):
        # Parse quaternions and camera positions
        l = nvm_poses[i+3].split(' ')
        f = '/'.join(
            l[0].rstrip('\n').split('/')[1:4]).replace('png', 'jpg')
        # print(f)
        if f in filenames:
            print("Found ", f)
            q = [float(x) for x in l[2:6]]
            c = [float(x) for x in l[6:9]]
            R = quaternion_matrix(q)[:3,:3]
            M = assemble_matrix(R, c)
            M[:3,3] = np.matmul(-M[:3,:3],M[:3,3])
            fname_to_pose[f] = [q, M]
    del nvm_poses
    return fname_to_pose

def load_triangulation_data(triangulation_file, filenames):
    """ Load triangulation data.

    Returns:
        A dictionary mapping reference image filenames to triangulation
        data (intrinsics matrix, distortion coefficients, 2D and 3D points)
    """
    assert os.path.isfile(triangulation_file)
    triangulation_data = np.load(triangulation_file, allow_pickle = True)
    filename_to_local_reconstruction = dict()
    for filename in filenames:
        key = key_converter(filename)
        if key in triangulation_data.files:
            local_reconstruction = triangulation_data[key].item()
            intrinsics, distortion_coefficients = _assemble_intrinsics(
                *local_reconstruction['K'].params)
            points_3D = local_reconstruction['points3D']
            points_2D = local_reconstruction['points2D']
            filename_to_local_reconstruction[filename] = \
                reconstruction_data(intrinsics, distortion_coefficients,
                    points_2D, points_3D)
    del triangulation_data
    return filename_to_local_reconstruction

def load_intrinsics(filenames):
    """Load query images intrinsics.

    For RobotCar, all images are rectified and have the same intrinsics.
    Returns:
        filename_to_intrinsics: A dictionary mapping a query filename to
            the intrinsics matrix and distortion coefficients.
    """
    rear_intrinsics = np.reshape(np.array(
        [400.0, 0.0, 508.222931,
        0.0, 400.0, 498.187378,
        0.0, 0.0, 1], dtype=np.float32), (3, 3))
    distortion_coefficients = np.array([0.0,0.0,0.0,0.0])
    intrinsics = [(rear_intrinsics, distortion_coefficients)
        for i in range(len(filenames))]
    filename_to_intrinsics = dict(
        zip(filenames, intrinsics))
    return filename_to_intrinsics