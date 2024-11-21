import numpy as np
import open3d as o3d
from scipy.interpolate import RegularGridInterpolator
import torch
import torch.nn.functional as F


def show_grid_points(points, voxel_volume):
    colors = voxel_volume[..., :3]
    # normalize?
    #min_col = np.min(colors) # axis=(0, 1, 2)
    #max_col = np.max(colors) # axis=(0, 1, 2)
    colors_normalized = colors #(colors - min_col) / (max_col - min_col)
    grid_points = np.array(np.meshgrid(points[0], points[1], points[2], indexing='ij')).reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_points)

    temp = colors_normalized.reshape(-1, 3)
    pcd.colors = o3d.utility.Vector3dVector(temp)
    return pcd


def show_spheres(xyz, interpolated_features):
    spheres = []
    N, _ = interpolated_features.shape
    for j in range(N):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        sphere.translate(xyz[j])
        sphere_color = interpolated_features[j]  # Take the first component
        sphere_color_normalized = sphere_color #(sphere_color - min_col) / (max_col - min_col)
        temp = sphere_color_normalized.reshape(-1, 3)
        sphere.paint_uniform_color(temp.squeeze())
        spheres += [sphere]
    return spheres


def trilinear_interpolation_batch(voxel_volume, xyz, origin, voxel_size):
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

    # normalize world positions xyz between -1 and 1
    xyz_norm = xyz
    xyz_norm -= origin
    xyz_norm /= torch.tensor([nx, ny, nz]) * voxel_size
    xyz_norm = 2 * xyz_norm - 1
    xyz_norm = xyz_norm.float()
    
    voxel_volume = voxel_volume.permute(0, 4, 3, 2, 1)  # (N, C, D_in, H_in, W_in)  # TODO: check if D=nx, H=ny, W=nz


    x = torch.linspace(0.0, nx*voxel_size, nx) + origin[0]
    y = torch.linspace(0.0, ny*voxel_size, ny) + origin[1]
    z = torch.linspace(0.0, nz*voxel_size, nz) + origin[2]
    points = (x, y, z) # x=(nx,) y=(ny,) z=(nz,)


    # Reshape xyz_norm to match grid_sample input
    sample = xyz_norm.view(B, N, 1, 1, 3)  # (B, N, 1, 1, 3)
    
    # Perform trilinear interpolation in a vectorized manner
    features = F.grid_sample(voxel_volume, sample, mode='bilinear', align_corners=True, padding_mode='border')
    
    # Reshape output to (B, N, C)
    features = features.view(B, C, N).permute(0, 2, 1)  # (B, C, N) -> (B, N, C)

    spheres = []
    for batch in range(B):
        pcd = show_grid_points(points, voxel_volume[batch].permute(3, 2, 1, 0))
        spheres = show_spheres(xyz[batch], features[batch])

        # zero box
        sphere_zero = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
        sphere_zero.translate((0, 0, 0))

        sphere_zero.paint_uniform_color((1.0, 0.0, 0.0))
        spheres += [sphere_zero]

        # origin box
        sphere_orig = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
        sphere_orig.translate((1, 0, -2))

        sphere_orig.paint_uniform_color((0.0, 1.0, 0.0))
        spheres += [sphere_orig]

        ###################
    o3d.visualization.draw_geometries([pcd]+spheres)

    return features


def trilinear_interpolation_batch_inefficient(voxel_volume, xyz, origin, voxel_size):
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

    # normalize world positions xyz between -1 and 1
    xyz_norm = xyz
    xyz_norm -= origin
    xyz_norm /= torch.tensor([nx, ny, nz]) * voxel_size
    xyz_norm = 2 * xyz_norm - 1
    xyz_norm = xyz_norm.float()
    
    voxel_volume = voxel_volume.permute(0, 4, 3, 2, 1)  # (N, C, D_in, H_in, W_in)  # TODO: check if D=nx, H=ny, W=nz


    x = torch.linspace(0.0, nx*voxel_size, nx) + origin[0]
    y = torch.linspace(0.0, ny*voxel_size, ny) + origin[1]
    z = torch.linspace(0.0, nz*voxel_size, nz) + origin[2]
    points = (x, y, z) # x=(nx,) y=(ny,) z=(nz,)


    spheres = []
    features = torch.empty(B, N, C)
    for batch in range(B):
        feat = torch.empty(N, C)
        for i in range(N):
            sample = torch.reshape(xyz_norm[batch, i], [1, 1, 1, 1, 3])  # (N, D_, H_, W_, 3)
            f = F.grid_sample(voxel_volume, sample, 'bilinear', align_corners=True)
            feat[i] = f.squeeze()
        features[batch] = feat

        pcd = show_grid_points(points, voxel_volume[batch].permute(3, 2, 1, 0))
        spheres = show_spheres(xyz[batch], feat)

        # zero box
        sphere_zero = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
        sphere_zero.translate((0, 0, 0))

        sphere_zero.paint_uniform_color((1.0, 0.0, 0.0))
        spheres += [sphere_zero]

        # origin box
        sphere_orig = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
        sphere_orig.translate((1, 0, -2))

        sphere_orig.paint_uniform_color((0.0, 1.0, 0.0))
        spheres += [sphere_orig]

        ###################
    o3d.visualization.draw_geometries([pcd]+spheres)

    return features


# advantage over pytorch grid_sample: scipy can extrapolate points outside of the volume
def trilinear_interpolation_batch_scipy(voxel_volume, xyz, origin, voxel_size):
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

    x = torch.linspace(0.0, nx*voxel_size, nx) + origin[0]
    y = torch.linspace(0.0, ny*voxel_size, ny) + origin[1]
    z = torch.linspace(0.0, nz*voxel_size, nz) + origin[2]
    points = (x, y, z) # x=(nx,) y=(ny,) z=(nz,)

    spheres = []
    features = torch.empty(B, N, C)
    for batch in range(B):

        interpolator = RegularGridInterpolator(points, voxel_volume[batch], bounds_error=False, fill_value=None) # extrapolate outside bounds
        interpolated_features = interpolator(xyz[batch])
        features[batch] = torch.from_numpy(interpolated_features)

        #### vizualize ####
        pcd = show_grid_points(points, voxel_volume[batch])
        spheres = show_spheres(xyz[batch], interpolated_features)

        # zero box
        sphere_zero = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
        sphere_zero.translate((0, 0, 0))

        sphere_zero.paint_uniform_color((1.0, 0.0, 0.0))
        spheres += [sphere_zero]

        # origin box
        sphere_orig = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
        sphere_orig.translate((1, 0, -2))

        sphere_orig.paint_uniform_color((0.0, 1.0, 0.0))
        spheres += [sphere_orig]

        ###################
    o3d.visualization.draw_geometries([pcd]+spheres)
    
    return features



origin = np.array([5, 0, 0])
voxel_size = 0.8
nx, ny, nz = (5, 6, 7)

fx = torch.linspace(0, 1, nx) # 0
fy = torch.linspace(0, 1, ny) # : 
fz = torch.linspace(0, 1, nz) # C-1
fX, fY, fZ = torch.meshgrid(fx, fy, fz, indexing='ij')
voxel_volume = torch.stack([fX, fY, fZ], axis=-1)
print("voxel_volume", voxel_volume.shape)


N = 500
epsilon = 5.0  # factor to outside points
#xyz = np.array([2.21, 3.12, 1.15])
xyz = torch.stack([
    torch.rand(N) * (nx+2*epsilon) * voxel_size,  # Random x-coordinates in range [0, nx * voxel_size]
    torch.rand(N) * (ny+2*epsilon) * voxel_size,  # Random y-coordinates in range [0, ny * voxel_size]
    torch.rand(N) * (nz+2*epsilon) * voxel_size  # Random z-coordinates in range [0, nz * voxel_size]
], dim=1)
xyz -= voxel_size * epsilon * np.ones(3)
xyz += origin
print("query", xyz.shape)

xyz = xyz.float()
#result = trilinear_interpolation(voxel_volume, xyz, origin, voxel_size)

result = trilinear_interpolation_batch(voxel_volume.unsqueeze(0), xyz.unsqueeze(0), origin, voxel_size)

#result2 = trilinear_interpolation_batch_scipy(voxel_volume.unsqueeze(0), xyz.unsqueeze(0), origin, voxel_size)

#print(result)
#print(result2)
#print(result == result2)
#print(result-result2)
