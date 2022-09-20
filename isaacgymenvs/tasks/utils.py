from isaacgym import gymapi
from pathlib import Path
from typing import Tuple

import math
import numpy as np
import torch


def arr2vec(arr: np.ndarray) -> gymapi.Vec3:
    """Convert array to gym Vec3"""
    arr = arr.reshape(-1)
    return gymapi.Vec3(arr[0], arr[1], arr[2])


def vec2arr(vec: gymapi.Vec3) -> np.ndarray:
    """Convert gym Vec3 to array"""
    return np.array([vec.x, vec.y, vec.z])


def arr2quat(arr: np.ndarray) -> gymapi.Quat:
    """Convert array to gym Quat"""
    return gymapi.Quat(arr[0], arr[1], arr[2], arr[3])


def quat2arr(quat: gymapi.Quat) -> np.ndarray:
    """Convert gym Quat to array"""
    return np.array([quat.x, quat.y, quat.z, quat.w])


def trans2tensor(pose: gymapi.Transform, device='cuda:0') -> torch.Tensor:
    """Convert pose expressed with gym Transform to pytorch tensor"""
    return torch.tensor([pose.p.x, pose.p.y, pose.p.z,
                         pose.r.x, pose.r.y, pose.r.z, pose.r.w], dtype=torch.float32, device=device)


def rot2quat(rot: np.ndarray) -> gymapi.Quat:
    """Convert a rotation matrix in SO(3) to gym Quat"""
    q = np.zeros((4,))
    q[0] = 0.5 * math.sqrt(1. + rot[0, 0] + rot[1, 1] + rot[2, 2])
    q[1:4] = 0.25 * q[0] * np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return gymapi.Quat(q[1], q[2], q[3], q[0])  # q = [x,y,z,w]


def quat2rot(q: gymapi.Quat) -> np.ndarray:
    """Convert gym Quat to rotation matrix in SO(3)"""
    rot = np.eye(3)
    q = np.array([q.w, q.x, q.y, q.z])
    rot[0, 0] = 2 * (q[0] ** 2 + q[1] ** 2) - 1
    rot[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
    rot[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])
    rot[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
    rot[1, 1] = 2 * (q[0] ** 2 + q[2] ** 2) - 1
    rot[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
    rot[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
    rot[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
    rot[2, 2] = 2 * (q[0] ** 2 + q[3] ** 2) - 1
    return rot


def trans(q: gymapi.Quat, p: gymapi.Vec3() = None) -> np.ndarray:
    """Create homogenous transformation from position and quaternion"""
    t = np.eye(4)
    # rotation matrix in SO(3)
    t[:3, :3] = quat2rot(q)
    # position vector
    if p is not None:
        t[:3, 3] = vec2arr(p)
    return t


def texture_mapping(s, t) -> Tuple[float, float, float, float, float, float, float]:
    """Map from texture space -> sphere space -> Euclidean space"""
    # texture coordinates -> sphere coordinates
    u = (-2 * np.pi * s) - np.pi  # horizontal - theta  [0,2pi]
    v = (-np.pi * t) - (np.pi / 2)  # vertical - phi  [0, phi]
    # cartesian coordinates
    x = math.cos(v) * math.cos(u)
    y = math.cos(v) * math.sin(u)
    z = math.sin(v)
    # sphere normal
    n1 = math.sin(v) * math.cos(u)
    n2 = math.sin(v) * math.sin(u)
    n3 = math.cos(v)
    # normal to quat
    angle = math.atan2(n1, n3)
    q0, q1, q2, q3 = (0., math.sin(angle/2), 0., math.cos(angle/2))
    # return x, y, z, n4, n1, n2, n3  (this partially works)
    return x, y, z, q0, q1, q2, q3


def trajectory_mapping(trajectory_2d: np.ndarray, sphere_rad=1.0) -> np.ndarray:
    """Map array of points texture space -> ... -> Euclidean space"""
    # coordinates in texture space
    s = trajectory_2d[:, 0]
    t = trajectory_2d[:, 1]
    # vector mapping function
    mapping_fun = np.vectorize(texture_mapping)
    # perform mapping to array
    xyz_n = mapping_fun(s, t)  # p(s,t) -> p(x,y,z,n)
    xyz_n = np.column_stack(xyz_n)
    xyz_n[:, 0:3] = xyz_n[:, 0:3] * sphere_rad
    return xyz_n


def get_textures(data_path='../data') -> np.ndarray:
    """Get array of texture filenames"""
    path = Path(f'./{data_path}/textures')
    textures = np.array([f.name for f in path.iterdir()], dtype=str)
    return textures


def get_trajectory(texture_id: str, data_path='../data') -> np.ndarray:
    """Get points in the path for a given texture id"""
    with open(f'{data_path}/paths/{texture_id}.npy', 'rb') as f:
        return np.load(f)
