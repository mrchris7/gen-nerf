from torch import nn
import functools
import numpy as np


def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        intrinsics = np.array([[float(num) for num in line.split()] for line in lines])
    return intrinsics


def camera_to_world(points_3d, trans):
    """
    Parameters:
        points_3d (n, 3): 3D points in camera coordinates
        trans (4, 4): transformation matrix (camera to world)
    Returns:
        (n, 3): 3D points in world coordinates
    """
    
    # convert to homogeneous coordinates
    points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # transform to world coordinates
    points_3d_world_hom = trans @ points_3d_hom.T
    points_3d_world = points_3d_world_hom[:3].T
    return points_3d_world


def get_norm_layer(norm_type="batch"):
    """
    Return a normalization layer
    Parameters:
        norm_type (str): the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError(f"normalization layer [{norm_type}] is not found")
    return norm_layer
