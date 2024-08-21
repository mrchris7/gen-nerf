import numpy as np
import open3d as o3d
import torch
from src.utils.visuals import display_color, display_depth


def create_camera_frustum(intrinsics, pose, near, far):
    """
    Create a frustum as an Open3D LineSet for visualization.
    
    Parameters:
        intrinsics (3, 3): camera intrinsic matrix
        pose (4, 4): camera pose in world coordinates (extrinsics)
        near (float): distance to the near plane
        far (float): distance to the far plane
        
    Returns:
        frustum: Open3D LineSet representing the camera frustum
    """
    
    # image plane dimensions in camera coordinates at near and far planes
    height_near = near * (1 / intrinsics[1, 1]) * intrinsics[1, 2] * 2
    width_near = near * (1 / intrinsics[0, 0]) * intrinsics[0, 2] * 2
    height_far = far * (1 / intrinsics[1, 1]) * intrinsics[1, 2] * 2
    width_far = far * (1 / intrinsics[0, 0]) * intrinsics[0, 2] * 2
    
    # frustum vertices in camera coordinates
    frustum_vertices = np.array([
        [width_near / 2, height_near / 2, near],
        [-width_near / 2, height_near / 2, near],
        [-width_near / 2, -height_near / 2, near],
        [width_near / 2, -height_near / 2, near],
        [width_far / 2, height_far / 2, far],
        [-width_far / 2, height_far / 2, far],
        [-width_far / 2, -height_far / 2, far],
        [width_far / 2, -height_far / 2, far]
    ])
    
    # convert to homogeneous coordinates
    frustum_vertices_h = np.hstack([frustum_vertices, np.ones((8, 1))])
    
    # transform vertices to world coordinates
    frustum_vertices_world = (pose @ frustum_vertices_h.T).T[:, :3]
    
    # define edges of the frustum (connecting the vertices)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # near plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # far plane
        [0, 4], [1, 5], [2, 6], [3, 7]   # connecting near and far planes
    ]
    
    # create LineSet for frustum
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(frustum_vertices_world)
    lines.lines = o3d.utility.Vector2iVector(edges)
    lines.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(edges))])  # Blue lines
    
    return lines


i = 0  # 0...4
data_folder = '/home/atuin/g101ea/g101ea13/debug/frustum_sampling/data/frustum_sampling'

# display images
color_file = f'{data_folder}/image_{i}.pt'
color_tensor = torch.load(color_file, map_location=torch.device('cpu'))
display_color(color_tensor)

depth_file = f'{data_folder}/depth_{i}.pt'
depth_tensor = torch.load(depth_file, map_location=torch.device('cpu'))
display_depth(depth_tensor)


# display frustum sampling
intrinsics_file = f'{data_folder}/intrinsics_{i}.pt'
intrinsics_tensor = torch.load(intrinsics_file, map_location=torch.device('cpu'))
intrinsics_np = intrinsics_tensor.squeeze().numpy()

pose_file = f'{data_folder}/pose_{i}.pt'
pose_tensor = torch.load(pose_file, map_location=torch.device('cpu'))
pose_np = pose_tensor.squeeze().numpy()

frustum = create_camera_frustum(intrinsics_np, pose_np, near=0.5, far=4.0)

# all points
point_cloud_file = f'{data_folder}/all_points_{i}.pt'
point_cloud_tensor = torch.load(point_cloud_file, map_location=torch.device('cpu'))
point_cloud_np = point_cloud_tensor.squeeze().numpy()
point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np)

# sampled points
sampled_point_cloud_file = f'{data_folder}/sampled_points_{i}.pt'
sampled_point_cloud_tensor = torch.load(sampled_point_cloud_file, map_location=torch.device('cpu'))
sampled_point_cloud_np = sampled_point_cloud_tensor.squeeze().numpy()
sampled_point_cloud_o3d = o3d.geometry.PointCloud()
sampled_point_cloud_o3d.points = o3d.utility.Vector3dVector(sampled_point_cloud_np)
colors_np1 = np.ones_like(sampled_point_cloud_np)
colors_np1[:, 0] = 1  # r
colors_np1[:, 1] = 0  # g
colors_np1[:, 2] = 0  # b
sampled_point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors_np1)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Frustum Sampling Visualization")
vis.add_geometry(sampled_point_cloud_o3d)
vis.add_geometry(frustum)
vis.add_geometry(point_cloud_o3d)

# set the point size of all points
render_option = vis.get_render_option()
render_option.point_size = 5.0

# set point size for sampled points
vis.update_geometry(sampled_point_cloud_o3d)
vis.get_render_option().point_size = 5.0

vis.run()
vis.destroy_window()
