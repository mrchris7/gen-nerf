import functools
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.interpolate import RegularGridInterpolator
from src.data.tsdf import coordinates
from src.utils.visuals import visualize_surface_and_connections


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


def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index


def normalize_coordinate(p, padding=0.1, plane='xz', encode=True):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


def normalize_3d_coordinate(p, padding=0.1):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    
    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def get_3d_points(depth_map, projection):
    """
    Parameters:
        depth_map (B, H, W)
        projection (B, 3, 4): world2image projection
    Returns:
        (B, H, W, 3): 3D points in world coordinates
    """
    B, H, W = depth_map.shape
    device = depth_map.device

    # generate a grid of coordinates
    u = torch.arange(0, W, device=device).view(1, -1).expand(H, -1)
    v = torch.arange(0, H, device=device).view(-1, 1).expand(-1, W)
    u = u.float().view(1, H, W)
    v = v.float().view(1, H, W)
        
    # create homogeneous coordinates for the image points
    ones = torch.ones_like(u)
    uv1 = torch.stack((u, v, ones), dim=-1)  # Shape: (1, H, W, 3)!
    uv1 = uv1.expand(B, -1, -1, -1)  # Shape: (B, H, W, 3)!
    
    # flatten uv1 and depth_maps
    uv1 = uv1.view(B, -1, 3)  # Shape: (B, N, 3)!

    depth_map_flat = depth_map.view(B, -1)  # Shape: (B, N)!

    # convert uv1 to 3D points using depth
    points_2d = uv1 * depth_map_flat.unsqueeze(-1)  # Shape: (B, N, 3)!

    # get projection matrix in shape by adding row [0, 0, 0, 1]: (B, 3, 4) -> (B, 4, 4)!
    # create the bottom row to be appended
    bottom_row = torch.tensor([0, 0, 0, 1], dtype=projection.dtype, device=projection.device)

    # repeat the bottom row for each matrix in the batch
    bottom_row = bottom_row.unsqueeze(0).repeat(B, 1).unsqueeze(1)  # Shape (B, 1, 4)!

    # concatenate the bottom row to the projection matrices
    projection_hom = torch.cat((projection, bottom_row), dim=1)  # Shape (B, 4, 4)!

    # inverse of the projection matrix
    inv_proj_matrix = torch.inverse(projection_hom)  # Shape: (B, 4, 4)!
    
    # convert 2D points to homogeneous coordinates
    points_2d_hom = torch.cat((points_2d, torch.ones_like(points_2d[..., :1])), dim=-1)  # Shape: (B, N, 4)

    # transform to 3D world coordinates
    points_3d_hom = torch.matmul(points_2d_hom, inv_proj_matrix.transpose(-1, -2))  # Shape: (B, N, 4)

    # normalize homogeneous coordinates
    points_3d = points_3d_hom[..., :3] / points_3d_hom[..., 3:4]  # Shape: (B, N, 3)

    assert(tuple(points_3d.shape) == (B, H*W, 3))
    
    points_3d = points_3d.reshape(B, H, W, 3)
    return points_3d


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        sampled_xyz: sparse pointcloud data [B, npoint, 3]
        centroids: mask of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1).to(device)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    # centroids = sampled pointcloud index, [B, npoint]
    sampled_xyz = xyz[torch.arange(B)[:, None], centroids]  # [B, npoint, 3]
    return sampled_xyz, centroids

# not used
def log_transform(x, shift=1):
    """ 
    Logarithmic scaling: rescales TSDF values to weight voxels near the surface 
    more than close to the truncation distance
    """
    return x.sign() * (1 + x.abs() / shift).log()


def smooth_log_transform(x, shift=1, beta=1):
    """
    Smooth logarithmic-like scaling without non-differentiable sign operations.
    Instead of `sign`, we use `tanh` for differentiability.
    
    Args:
    - x: Tensor, TSDF values
    - shift: A small positive constant to avoid singularities
    - beta: Controls the smoothness of the transformation
    
    Returns:
    - Transformed TSDF values with smooth gradients.
    """
    # smooth approximation to sign using tanh
    return torch.tanh(x) * torch.nn.functional.softplus(x.abs() / shift, beta=beta)


def gaussian_kernel(kernel_size, sigma, device):
    """Generates a 2D Gaussian kernel."""
    # create a 1D tensor with equally spaced values centered at 0
    x = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss_1d = torch.exp(-x.pow(2) / (2 * sigma**2))
    gauss_1d /= gauss_1d.sum()  # normalize

    # generate a 2D kernel
    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    # additional dim for compatibility with conv2d
    gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)
    gauss_2d = gauss_2d.to(device)
    return gauss_2d

def apply_gaussian_smoothing(image, kernel_size, sigma):
    """
    Apply Gaussian smoothing to a batch of images.
    
    Args:
    - image: A tensor of shape (B, C, H, W)
    - kernel_size: The size of the Gaussian kernel
    - sigma: The standard deviation of the Gaussian distribution
    
    Returns:
    - Smoothed image (B, C, H, W)
    """
    B, C, H, W = image.shape
    
    # create a gaussian kernel (1, 1, kernel_size, kernel_size)
    kernel = gaussian_kernel(kernel_size, sigma, device=image.device)
    
    # repeat the kernel for each channel in the batch
    kernel = kernel.repeat(C, 1, 1, 1)  # (C, 1, kernel_size, kernel_size)
    
    # apply the filter to each channel
    # 'padding='same' to keep the output size the same as input
    smoothed_image = F.conv2d(image, kernel, padding=kernel_size // 2, groups=C)
    
    return smoothed_image

'''
def unproj_map(width, height, f, c=None, device="cpu"):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - float(c[1]),
        torch.arange(width, dtype=torch.float32) - float(c[0]),
        indexing="ij",
    )
    X = X.to(device=device) / float(f[0])
    Y = Y.to(device=device) / float(f[1])
    Z = torch.ones_like(X)
    unproj = torch.stack((X, -Y, -Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj


def gen_rays(poses, width, height, focal, z_near, z_far, c=None):
    """
    Generate camera rays
    :return (B, H, W, 8)
    """
    num_images = poses.shape[0]
    device = poses.device
    if len(focal.shape) > 0 and focal.shape[-1] > 1:
        cam_unproj_map = []
        for img_idx in range(num_images):
            temp_unproj_map = unproj_map(
                width, height, focal.squeeze()[img_idx], c=c.squeeze()[img_idx, :], device=device
            )
            cam_unproj_map.append(temp_unproj_map)
        cam_unproj_map = torch.stack(cam_unproj_map, dim=0)
    else:
        cam_unproj_map = (
            unproj_map(width, height, focal.squeeze(), c=c, device=device).unsqueeze(0).repeat(num_images, 1, 1, 1)
        )
    cam_centers = poses[:, None, None, :3, 3].expand(-1, height, width, -1)
    cam_raydir = torch.matmul(poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1))[:, :, :, :, 0]

    cam_nears = torch.tensor(z_near, device=device).view(1, 1, 1, 1).expand(num_images, height, width, -1)
    cam_fars = torch.tensor(z_far, device=device).view(1, 1, 1, 1).expand(num_images, height, width, -1)
    return torch.cat((cam_centers, cam_raydir, cam_nears, cam_fars), dim=-1)  # (B, H, W, 8)
'''

def sample_pixels(B, H, W, num_samples, device):
    # randomly sample pixel coordinates
    h_idxs = torch.randint(0, H, (B, num_samples), device=device) # [B, num_samples]
    w_idxs = torch.randint(0, W, (B, num_samples), device=device) # [B, num_samples]
    b_idxs = torch.arange(B, device=device).unsqueeze(1)
    return b_idxs, h_idxs, w_idxs


def sample_valid_depth_pixels(depth, num_samples):
    # randomly sample valid pixel coordinates from a depth map
    B, H, W = depth.shape
    device = depth.device
    b_idxs = torch.arange(B, device=device).unsqueeze(1)
    
    
    idxs = []
    for b in range(B):
        valid_indices = torch.argwhere(depth[b] != 0)  # (num_valid_samples, 2)
        num_valid_samples = valid_indices.shape[0]

        if num_valid_samples < num_samples:
            raise ValueError("Not enough non-zero depth pixels to sample from.")
                
        # extract randomly chosen and valid depth map indices
        selected_indices = torch.randperm(num_valid_samples, device=device)[:num_samples] # (num_valid_samples, 2)
        rand_valid_indices = valid_indices[selected_indices] # (num_samples, 2)
        idxs.append(rand_valid_indices)

    idxs = torch.stack(idxs, dim=0)  # (B, num_samples, 2)
    h_idxs = idxs[..., 0]  # (B, num_samples)
    w_idxs = idxs[..., 1]  # (B, num_samples)
    return b_idxs, h_idxs, w_idxs


def sample_valid_pixels(depth, normals, num_samples):
    # randomly sample valid pixel coordinates from a depth map
    B, H, W = depth.shape
    device = depth.device

    idxs = []
    for b in range(B):
        valid_depth = depth[b] != 0  # (H, W)
        valid_normals = ~torch.isnan(normals[b]).any(dim=2)  # (H, W)                
        valid = torch.logical_and(valid_depth, valid_normals)  # (H, W)        
        
        valid_idxs = torch.argwhere(valid)  # (num_valid_idxs, 2)
        num_valid_idxs = valid_idxs.shape[0]
        
        if num_valid_idxs < num_samples:
            raise ValueError("Not enough valid pixels to sample from.")
                
        # extract randomly chosen and valid depth map indices
        selected_idxs = torch.randperm(num_valid_idxs, device=device)[:num_samples] # (num_samples, 2)
        rand_valid_idxs = valid_idxs[selected_idxs] # (num_samples, 2)
        idxs.append(rand_valid_idxs)

    idxs = torch.stack(idxs, dim=0)  # (B, num_samples, 2)
    h_idxs = idxs[..., 0]  # (B, num_samples)
    w_idxs = idxs[..., 1]  # (B, num_samples)
    b_idxs = torch.arange(B, device=device).unsqueeze(1)  # (B, 1)

    return b_idxs, h_idxs, w_idxs
        

def sample_points_from_bounding_box(xyz, num_samples):
    """
    Calculate the bounding box of the point cloud and sample points from the volume.
    
    Parameters:
        xyz (B, N, 3)
        num_samples int
    Returns:
        (B, num_samples, 3)
    """ 
    min_bounds = np.min(xyz, axis=1, keepdims=True)  # (B, 1, 3)
    max_bounds = np.max(xyz, axis=1, keepdims=True)  # (B, 1, 3)

    # sample points uniformly within the bounding box for each point cloud
    sampled_xyz = np.random.uniform(low=min_bounds, high=max_bounds, size=(xyz.shape[0], num_samples, 3))

    return sampled_xyz


def sample_points_in_frustum(h_idxs, w_idxs, intrinsics, pose, min_dist, max_dist):
    """
    Sample points within the camera's view frustum in world coordinates for a batch of cameras.
    
    Parameters:
        intrinsics (B, 3, 3): batch of camera intrinsic matrices
        poses (B, 4, 4): batch of camera poses in world coordinates (extrinsics)
        num_samples (int): number of points to sample for each camera
        min_dist (float): minimum distance from the camera
        max_dist (float): maximum distance from the camera
        
    Returns:
        xyz_world (B, num_samples, 3): sampled points in world coordinates for each camera
    """
    device = intrinsics.device
    B, num_samples = h_idxs.shape

    # sample depth values between min_dist and max_dist for each camera
    # pow = sqrt: sample more points in the distance (for equal distribution in frustum)
    z = torch.pow(torch.rand(B, num_samples, device=device), 1.0/2.0) * (max_dist - min_dist) + min_dist    

    # convert 2D pixel coordinates to normalized image coordinates for each camera
    w_norm = (w_idxs - intrinsics[:, 0, 2].unsqueeze(-1)) / intrinsics[:, 0, 0].unsqueeze(-1)  # (u - cx) / fx
    h_norm = (h_idxs - intrinsics[:, 1, 2].unsqueeze(-1)) / intrinsics[:, 1, 1].unsqueeze(-1)  # (v - cy) / fy

    # compute the corresponding 3D coordinates in the camera space
    x = w_norm * z
    y = h_norm * z

    xyz_camera = torch.stack((x, y, z), dim=-1)  # (B, num_samples, 3)

    # convert points to homogeneous coordinates
    xyz_camera_hom = torch.cat((xyz_camera, torch.ones(B, num_samples, 1, device=device)), dim=-1)  # (B, num_samples, 4)

    # transform points from camera coordinates to world coordinates using the pose matrix
    xyz_world_hom = torch.bmm(pose, xyz_camera_hom.permute(0, 2, 1)).permute(0, 2, 1)  # (B, num_samples, 4)

    # convert back from homogeneous coordinates to cartesian coordinates
    xyz_world = xyz_world_hom[:, :, :3] / xyz_world_hom[:, :, 3:] # (B, num_samples, 3)

    return xyz_world, z


def sample_points_on_rays(h_idxs, w_idxs, depths, intrinsics, poses, N, M, delta, min_dist, sigma):
    """
    Sample points within the camera's view frustum in world coordinates for a batch of cameras using rays.
    Implementation of iSDF: https://arxiv.org/abs/2204.02296
    
    Parameters:
        h_idxs (B, num_samples): height indices of sampled pixels for each camera
        w_idxs (B, num_samples): width indices of sampled pixels for each camera
        depths (B, num_samples): depth of sampled pixels for each camera
        intrinsics (B, 3, 3): batch of camera intrinsic matrices
        poses (B, 4, 4): batch of camera poses in world coordinates (extrinsics)
        N: number of stratified samples
        M: number of samples around the surface
        delta: distance behind surface
        min_dist: minimum distance for stratified sampling
        sigma: standard deviation for Gaussian samples
        
    Returns:
        xyz_world     (B, num_samples, (1+N+M), 3): sampled points in world coordinates for each camera
        z_world       (B, num_samples, (1+N+M)   ): sampled depths values corresponding to every point
    """
    B, num_samples = depths.shape
    device = intrinsics.device
    
    # sample depth values
    sampled_depths_list = []
    for b in range(B):
        D = depths[b]  # depth at sampled pixels
        
        # N stratified samples in range [min_dist, D + delta] for each sample
        stratified_depths = torch.empty(num_samples, N, device=device)  # (num_samples, N)
        for i in range(num_samples):
            stratified_depths[i] = torch.linspace(min_dist, D[i] + delta, N, device=device)  # (N,)
        
        # M samples from gaussian distribution around the surface depth
        D_extended = D.unsqueeze(-1).expand(num_samples, M).to(device) # (num_samples, M)
        sigma_extended = sigma * torch.ones((num_samples, M), device=device)  # (num_samples, M)
        gaussian_depths = torch.normal(D_extended, sigma_extended)  # (num_samples, M)
        
        # 1 surface sample
        surface_depth = D.unsqueeze(-1)  # (num_samples, 1)
        
        # combine samples N+M+1
        sampled_depths = torch.cat((surface_depth, stratified_depths, gaussian_depths), dim=1)  # (num_samples, 1+N+M)
        sampled_depths_list.append(sampled_depths)

    # stack the results from all cameras/batches
    z_mat = torch.stack(sampled_depths_list)  # (B, num_samples, N+M+1)

    # convert 2D pixel coordinates to normalized image coordinates for each camera
    w_norm = (w_idxs - intrinsics[:, 0, 2].unsqueeze(-1)) / intrinsics[:, 0, 0].unsqueeze(-1)  # (u - cx) / fx
    h_norm = (h_idxs - intrinsics[:, 1, 2].unsqueeze(-1)) / intrinsics[:, 1, 1].unsqueeze(-1)  # (v - cy) / fy

    # repeat 2nd dimension (all ray points correspond to same (u,v)-pixel)
    w_repeated = w_norm.unsqueeze(-1).repeat(1, 1, z_mat.shape[2])  # (B, num_samples, 1+N+M)
    h_repeated = h_norm.unsqueeze(-1).repeat(1, 1, z_mat.shape[2])  # (B, num_samples, 1+N+M)

    # compute the corresponding 3D coordinates in the camera space
    x_mat = w_repeated * z_mat
    y_mat = h_repeated * z_mat

    # (B, num_samples, N+M+1) -> (B, num_samples*(1+N+M))
    num_points = num_samples * (1+N+M)
    x = x_mat.view(B, num_points)
    y = y_mat.view(B, num_points)
    z = z_mat.view(B, num_points)

    xyz_camera = torch.stack((x, y, z), dim=-1)  # (B, num_points, 3)

    # convert points to homogeneous coordinates
    xyz_camera_hom = torch.cat((xyz_camera, torch.ones(B, num_points, 1, device=device)), dim=-1)  # (B, num_points, 4)

    # transform from camera to world coordinates
    xyz_world_hom = torch.bmm(poses, xyz_camera_hom.permute(0, 2, 1)).permute(0, 2, 1)  # (B, num_points, 4)

    # convert back to cartesian coordinates
    xyz_world = xyz_world_hom[:, :, :3] / xyz_world_hom[:, :, 3:] # (B, num_points, 3)

    # extract samples at surface
    xyz_world_reshaped = xyz_world.reshape(B, num_samples, -1, 3)
    #surface_world = xyz_world_reshaped[:, :, 0:1, :]
    #surface_world = surface_world.squeeze(2)  # (B, num_samples, 3)
    return xyz_world_reshaped, z_mat


def bounds_pc(pc, z_vals, depth_sample, do_grad=True):
    """
    Parameters:
        pc (n_rays, 1+N+M, 3): point cloud
        z_val (n_rays, 1+N+M): depth values
        depth_sample (n_rays): depth value samples at sampled rays
    Returns:
        bounds (n_rays, 1+N+M): Signed distances from points to closest surface points
        grad (n_rays, N+M, 3): Gradient vectors for points
    """

    # from loss.py of iSDF
    with torch.set_grad_enabled(False):
        surf_pc = pc[:, 0]
        diff = pc[:, :, None] - surf_pc
        dists = diff.norm(dim=-1)
        dists, closest_ixs = dists.min(axis=-1)
        behind_surf = z_vals > depth_sample[:, None]
        dists[behind_surf] *= -1
        bounds = dists

        grad = None
        if do_grad:
            ix1 = torch.arange(diff.shape[0])[:, None]
            ix1 = ix1.repeat(1, diff.shape[1])
            ix2 = torch.arange(diff.shape[1])[None, :]
            ix2 = ix2.repeat(diff.shape[0], 1)
            grad = diff[ix1, ix2, closest_ixs]
            grad = grad[:, 1:]
            grad = grad / grad.norm(dim=-1)[..., None]

            # flip grad vectors behind the surf
            grad[behind_surf[:, 1:]] *= -1

        #visualize_surface_and_connections(pc, surf_pc, closest_ixs)
        
    return bounds, grad

def bounds_pc_batch(pc, z_vals, depth_sample, do_grad=True):
    """
    Adapted from loss.py of iSDF to work on batches.
    Parameters:
        pc (B, n_rays, 1+N+M), 3): point cloud
        z_val (B, n_rays, 1+N+M): depth values
        depth_sample (B, n_rays): depth value samples at sampled rays
    Returns:
        bounds (B, n_rays, 1+N+M   ): Signed distances from points to closest surface points
        grad   (B, n_rays,   N+M, 3): Gradient vectors for points
    """
    with torch.set_grad_enabled(False):
        B = pc.shape[0]
        surf_pc = pc[:, :, 0]  # (B, n_rays, 3) surface points

        # calculate differences
        diff = [pc[b, :, :, None] - surf_pc[b] for b in range(B)]      
        diff = torch.stack(diff, dim=0)  # (B, n_rays, (1+N+M), n_rays, 3)

        # calculate distances
        dists = diff.norm(dim=-1)  # (B, n_rays, (1+N+M), n_rays)
        min_dists, closest_ixs = dists.min(dim=-1)  # (B, n_rays, (1+N+M)), (B, n_rays, (1+N+M))

        # identify points behind the surface
        behind_surf = z_vals > depth_sample[:, :, None]  # (B, n_rays, (1+N+M))

        # modify distances for points behind the surface
        min_dists[behind_surf] *= -1
        
        bounds = min_dists  # (B, n_rays, (1+N+M))

        grad = None
        if do_grad:
            ix0 = torch.arange(diff.shape[0])[:, None, None] # (B, 1, 1)
            ix0 = ix0.repeat(1, diff.shape[1], diff.shape[2]) # (B, n_rays, (1+N+M))
            ix1 = torch.arange(diff.shape[1])[None, :, None] # (1, n_rays, 1)
            ix1 = ix1.repeat(diff.shape[0], 1, diff.shape[2]) # (B, n_rays, (1+N+M))            
            ix2 = torch.arange(diff.shape[2])[None, None, :] # (1, 1, (1+N+M))
            ix2 = ix2.repeat(diff.shape[0], diff.shape[1], 1) # (B, n_rays, (1+N+M))
            
            # select differences for closest points
            grad = diff[ix0, ix1, ix2, closest_ixs] # (B, n_rays, (1+N+M), 3)
            
            # exclude gradients for the surface point
            grad = grad[:, :, 1:] # (B, n_rays, (N+M), 3)

            # normalize gradient
            grad = grad / grad.norm(dim=-1, keepdim=True)  # (B, n_rays, (N+M), 3)
            
            # flip gradients for points behind the surface
            grad[behind_surf[:, :, 1:]] *= -1

    #visualize_surface_and_connections(pc[0], surf_pc[0], closest_ixs[0])
    return bounds, grad

def calculate_grad(inputs, outputs):

    # compute the gradient of the output tsdfs with respect to the input points
    grad_outputs = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    gradient = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (B, N, 3)
    return gradient


# for renderer
def get_sphere_intersection(cam_loc, ray_directions, r = 1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays

    n_imgs, n_pix, _ = ray_directions.shape
    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0
    
    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect


# for renderer
def image_points_to_world(image_points, camera_mat, world_mat, scale_mat,
                          invert=True):
    ''' Transforms points on image plane to world coordinates.

    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.

    Args:
        image_points (tensor): image points tensor of size B x N x 2
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    device = image_points.device
    d_image = torch.ones(batch_size, n_pts, 1).to(device)
    return transform_to_world(image_points, d_image, camera_mat, world_mat,
                              scale_mat, invert=invert)


# for renderer
def transform_to_world(pixels, depth, camera_mat, world_mat, scale_mat,
                       invert=True):
    ''' Transforms pixel positions p with given depth value d to world coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)

    # Invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    p_world = scale_mat @ world_mat @ camera_mat @ pixels

    # Transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_world = p_world.numpy()
    return p_world


# for renderer
def origin_to_world(n_points, camera_mat, world_mat, scale_mat, invert=True):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: true)
    '''
    batch_size = camera_mat.shape[0]
    device = camera_mat.device

    # Create origin in homogen coordinates
    p = torch.zeros(batch_size, 4, n_points).to(device)
    p[:, -1] = 1.

    # Invert matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Apply transformation
    p_world = scale_mat @ world_mat @ camera_mat @ p

    # Transform points back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world


# for renderer
def get_mask(tensor):
    ''' Returns mask of non-illegal values for tensor.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
    '''
    tensor, is_numpy = to_pytorch(tensor, True)
    mask = ((abs(tensor) != np.inf) & (torch.isnan(tensor) == False))
    mask = mask.bool()
    if is_numpy:
        mask = mask.numpy()

    return mask


# for renderer
def sample_patch_points(batch_size, n_points, patch_size=1,
                        image_resolution=(128, 128), 
                        sensor_size=np.array([[-1, 1],[-1, 1]]), 
                        continuous=True):
    ''' Returns sampled points in the range of sensorsize.

    Args:
        batch_size (int): required batch size
        n_points (int): number of points to sample
        patch_size (int): size of patch; if > 1, patches of size patch_size
            are sampled instead of individual points
        image_resolution (tuple): image resolution (required for calculating
            the pixel distances)
        continuous (bool): whether to sample continuously or only on pixel
            locations
    '''
    assert(patch_size > 0)

    n_patches = int(n_points)

    if continuous:
        p = torch.rand(batch_size, n_patches, 2)  # [0, 1]
    else:
        px = torch.randint(0, image_resolution[1], size=(
            batch_size, n_patches, 1)).float() 
        py = torch.randint(0, image_resolution[0], size=(
            batch_size, n_patches, 1)).float() 
        p = torch.cat([px, py], dim=-1)

    p = p.view(batch_size, n_patches, 1, 2) 
    p = p.view(batch_size, -1, 2)
    pix = p.clone()

    p[:,:,0] *= (sensor_size[1,1] - sensor_size[1,0]) / (image_resolution[1] - 1)
    p[:,:,1] *= (sensor_size[0,1] - sensor_size[0,0])  / (image_resolution[0] - 1)
    p[:,:,0] += sensor_size[1,0] 
    p[:,:,1] += sensor_size[0,0]

    assert(p.max() <= sensor_size.max())
    assert(p.min() >= sensor_size.min())
    assert(pix.max() < max(image_resolution))
    assert(pix.min() >= 0)

    return p, pix


# for renderer
def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                  subsample_to=None):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h), indexing='ij')
    pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = (image_range[1] - image_range[0])/ 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and
            subsample_to < n_points):
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                               replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    return pixel_locations, pixel_scaled


# for renderer
def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True

    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor


# for renderer
def combine_interleaved(t, inner_dims=(1,), agg_type="average"):
    if len(inner_dims) == 1 and inner_dims[0] == 1:
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t

def add_dicts(dict1, dict2):
    if len(dict1.keys()) == 0:
        return dict2
    
    if len(dict2.keys()) == 0:
        return dict1

    result = {}
    for key in dict1:
        result[key] = dict1.get(key, 0) + dict2.get(key, 0)
    return result

def get_grid_coordinates(nx, ny, nz, volume_size, origin, device):
    x = torch.linspace(0, volume_size[0], nx, device=device)
    y = torch.linspace(0, volume_size[1], ny, device=device)
    z = torch.linspace(0, volume_size[2], nz, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

    # stack the grid coordinates and reshape to match input shape (B, N, 3)
    grid_xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (nx, ny, nz, 3)
    #grid_xyz = grid_xyz + origin.to(device)
    return grid_xyz

def get_corner_coordinates(volume_size, origin, device):
    # i.e. volume_size = [ 8.   10.    2.56]
    x_values = torch.tensor([0, volume_size[0]], device=device)
    y_values = torch.tensor([0, volume_size[1]], device=device)
    z_values = torch.tensor([0, volume_size[2]], device=device)

    corner_xyz = torch.cartesian_prod(x_values, y_values, z_values)
    #corner_xyz = corner_xyz + origin.to(device)
    return corner_xyz


def backproject(voxel_dim, voxel_size, origin, projection, features):
    """ Takes 2d features and fills them along rays in a 3d volume

    This function implements eqs. 1,2 in https://arxiv.org/pdf/2003.10432.pdf
    Each pixel in a feature image corresponds to a ray in 3d.
    We fill all the voxels along the ray with that pixel's features.

    Args:
        voxel_dim: size of voxel volume to construct (nx, ny, nz)
        voxel_size: metric size of each voxel (ex: .04m)
        origin (1, 3): origin of the voxel volume (xyz position of voxel (0,0,0))
        projection (B, 4, 3): projection matrices (intrinsics@extrinsics)
        features (B, C, H, W): 2d feature tensor to be backprojected into 3d

    Returns:
        volume (B, C, nx, ny, nz): 3d feature volume
        valid (B, 1, nx, ny, nz): boolean volume, each voxel contains a 1 if it projects to a
                                  pixel and 0 otherwise (not in view frustrum of the camera)
    """

    B = features.size(0)
    C = features.size(1)
    #print("***d_vol:", C)
    device = features.device
    nx, ny, nz = voxel_dim

    coords = coordinates(voxel_dim, device).unsqueeze(0).expand(B,-1,-1) # (B, 3, H, W, D)
    world = coords.type_as(projection) * voxel_size + origin.to(device).unsqueeze(2)
    world = torch.cat((world, torch.ones_like(world[:,:1]) ), dim=1)
    
    camera = torch.bmm(projection, world)
    px = (camera[:,0,:]/camera[:,2,:]).round().type(torch.long)
    py = (camera[:,1,:]/camera[:,2,:]).round().type(torch.long)
    pz = camera[:,2,:]

    # voxels in view frustrum
    height, width = features.size()[2:]
    valid = (px >= 0) & (py >= 0) & (px < width) & (py < height) & (pz>0) # bxhwd

    # put features in volume
    volume = torch.zeros(B, C, nx*ny*nz, dtype=features.dtype, 
                         device=device)
    for b in range(B):
        volume[b,:,valid[b]] = features[b,:,py[b,valid[b]], px[b,valid[b]]]

    volume = volume.view(B, C, nx, ny, nz)
    valid = valid.view(B, 1, nx, ny, nz)

    return volume, valid


def trilinear_interpolation(voxel_volume, xyz, origin, voxel_size, mode='bilinear'):
    """
    Perform trilinear interpolation to map 3D world points to features in the voxel volume.
    
    Args:
        voxel_volume (B, nx, ny, nz, C): voxel volume
        xyz (B, N, 3): 3D world points
        origin (3,): world coordinates of voxel (0, 0, 0)
        voxel_size: size of each voxel
    
    Returns:
        features (B, N, C): interpolated features
    """

    B, nx, ny, nz, C = voxel_volume.shape
    N = xyz.shape[1]
    device = voxel_volume.device

    # normalize world positions xyz between -1 and 1
    xyz = xyz - origin.to(device)
    xyz = xyz / (torch.tensor([nx, ny, nz], device=device) * voxel_size)
    xyz = 2 * xyz - 1
    xyz = xyz.float()
    
    voxel_volume = voxel_volume.permute(0, 4, 3, 2, 1)  # (B, nx, ny, nz, C) -> (N, C, D, H, W)
    samples = xyz.view(B, N, 1, 1, 3)  # (B, N, 1, 1, 3)
    ''' alternative (may break grad flow?)
    xyz_norm = xyz.clone()
    xyz_norm -= origin.to(device)
    xyz_norm /= torch.tensor([nx, ny, nz], device=device) * voxel_size
    xyz_norm = 2 * xyz_norm - 1
    xyz_norm = xyz_norm.float()
    
    voxel_volume = voxel_volume.permute(0, 4, 3, 2, 1)  # (B, nx, ny, nz, C) -> (N, C, D, H, W)
    samples = xyz_norm.view(B, N, 1, 1, 3)  # (B, N, 1, 1, 3)
    '''
    features = F.grid_sample(voxel_volume, samples, mode=mode, align_corners=True, padding_mode='border')
    #features = grid_sample_3d(voxel_volume, samples)
    #print("features_old", features_old.shape)
    #print("features_new", features_new.shape)
    #assert(torch.allclose(features_old, features_new, atol=1e-3, rtol=1e-4))
    features = features.view(B, C, N).permute(0, 2, 1)  # (B, C, N) -> (B, N, C)
    
    return features


def trilinear_interpolation_suboptimal(voxel_volume, xyz, origin, voxel_size):
    """
    Perform trilinear interpolation to map 3D world points to features in the voxel volume.
    
    Args:
        voxel_volume (B, nx, ny, nz, C): voxel volume
        xyz (B, N, 3): 3D world points
        origin (3,): world coordinates of voxel (0, 0, 0)
        voxel_size: size of each voxel
    
    Returns:
        features (B, N, C): interpolated features
    """
    B, nx, ny, nz, C = voxel_volume.shape
    N = xyz.shape[1]
    device = voxel_volume.device

    # normalize world positions xyz between -1 and 1
    xyz = xyz - origin.to(device)
    xyz = xyz / torch.tensor([nx, ny, nz]).to(device) * voxel_size
    xyz = 2 * xyz - 1
    xyz = xyz.float()
    
    voxel_volume = voxel_volume.permute(0, 4, 3, 2, 1)  # (N, C, D_in, H_in, W_in)  # TODO: check if D=nx, H=ny, W=nz

    features = torch.empty(B, N, C).to(device)
    for batch in range(B):
        feat = torch.empty(N, C).to(device)
        for i in range(N):
            sample = torch.reshape(xyz[batch, i], [1, 1, 1, 1, 3])  # (N, D_, H_, W_, 3)
            f = F.grid_sample(voxel_volume, sample, 'bilinear', align_corners=True)
            feat[i] = f.squeeze()
        features[batch] = feat

    return features


'''
# use this if spatial net is used also -> else: CUDA OUT OF MEMORY
def trilinear_interpolation(voxel_volume, xyz, origin, voxel_size):
    """
    Perform trilinear interpolation to map 3D world points to features in the voxel volume.
    
    Args:
        voxel_volume (B, nx, ny, nz, C): voxel volume
        xyz (B, N, 3): 3D world points
        origin (3,): world coordinates of voxel (0, 0, 0)
        voxel_size: size of each voxel
    
    Returns:
        features (B, N, C): interpolated features
    """
    device = voxel_volume.device
    B, nx, ny, nz, C = voxel_volume.shape
    N = xyz.shape[1]
    
    x = torch.linspace(0.0, nx*voxel_size, nx) + origin[0]
    y = torch.linspace(0.0, ny*voxel_size, ny) + origin[1]
    z = torch.linspace(0.0, nz*voxel_size, nz) + origin[2]
    points = (x, y, z) # x=(nx,) y=(ny,) z=(nz,)

    features = torch.empty(B, N, C, device=device)
    for batch in range(B):

        interpolator = RegularGridInterpolator(points, voxel_volume[batch].detach().cpu(),
                                               bounds_error=False, fill_value=None) # extrapolate outside bounds
        interpolated_features = interpolator(xyz[batch].detach().cpu())
        features[batch] = torch.from_numpy(interpolated_features).to(device)
    
    return features
'''

def grid_sample_2d(image, optical):
    """
    Alternative for pytorch's 2D case of
    F.grid_sample(image, optical, padding_mode='border', align_corners=True, mode='bilinear")
    that has an implementation for the backward pass

    Source: https://github.com/pytorch/pytorch/issues/34704
    """
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


def grid_sample_3d(image, optical):
    """
    Alternative for pytorch's 3D case of
    F.grid_sample(image, optical, padding_mode='border', align_corners=True, mode='bilinear")
    that has an implementation for the backward pass

    Source: https://github.com/pytorch/pytorch/issues/34704
    """
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    iz = ((iz + 1) / 2) * (ID - 1);
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix);
        iy_tnw = torch.floor(iy);
        iz_tnw = torch.floor(iz);

        ix_tne = ix_tnw + 1;
        iy_tne = iy_tnw;
        iz_tne = iz_tnw;

        ix_tsw = ix_tnw;
        iy_tsw = iy_tnw + 1;
        iz_tsw = iz_tnw;

        ix_tse = ix_tnw + 1;
        iy_tse = iy_tnw + 1;
        iz_tse = iz_tnw;

        ix_bnw = ix_tnw;
        iy_bnw = iy_tnw;
        iz_bnw = iz_tnw + 1;

        ix_bne = ix_tnw + 1;
        iy_bne = iy_tnw;
        iz_bne = iz_tnw + 1;

        ix_bsw = ix_tnw;
        iy_bsw = iy_tnw + 1;
        iz_bsw = iz_tnw + 1;

        ix_bse = ix_tnw + 1;
        iy_bse = iy_tnw + 1;
        iz_bse = iz_tnw + 1;

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);


    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.reshape(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val


# adapted from https://github.com/wkentaro/morefusion/blob/main/morefusion/geometry/estimate_pointcloud_normals.py
def estimate_pointcloud_normals(points):
    # These lookups denote yx offsets from the anchor point for 8 surrounding
    # directions from the anchor A depicted below.
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 | A | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    assert points.shape[2] == 3

    d = 2
    H, W = points.shape[:2]
    points = torch.nn.functional.pad(
        points,
        pad=(0, 0, d, d, d, d),
        mode="constant",
        value=float('nan'),
    )

    lookups = torch.tensor(
        [(-d, 0), (-d, d), (0, d), (d, d), (d, 0), (d, -d), (0, -d), (-d, -d)]
    ).to(points.device)

    j, i = torch.meshgrid(torch.arange(W), torch.arange(H))
    i = i.transpose(0, 1).to(points.device)
    j = j.transpose(0, 1).to(points.device)
    k = torch.arange(8).to(points.device)

    i1 = i + d
    j1 = j + d
    points1 = points[i1, j1]

    lookup = lookups[k]
    i2 = i1[None, :, :] + lookup[:, 0, None, None]
    j2 = j1[None, :, :] + lookup[:, 1, None, None]
    points2 = points[i2, j2]

    lookup = lookups[(k + 2) % 8]
    i3 = i1[None, :, :] + lookup[:, 0, None, None]
    j3 = j1[None, :, :] + lookup[:, 1, None, None]
    points3 = points[i3, j3]
    
    diff = torch.linalg.norm(points2 - points1, dim=3) + torch.linalg.norm(
        points3 - points1, dim=3
    )
    diff[torch.isnan(diff)] = float('inf')
    indices = torch.argmin(diff, dim=0)

    normals = torch.cross(
        points2[indices, i, j] - points1[i, j],
        points3[indices, i, j] - points1[i, j],
    )
    normals /= torch.linalg.norm(normals, dim=2, keepdims=True)
    return normals
