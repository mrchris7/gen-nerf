import numpy as np
import open3d as o3d
import torch



# CONFIG:
base_folder = '../../logs/train/runs/2024-11-21_01-21-54/local'
mesh_color = [0.75, 0.75, 0.75]
show_pred_mesh = True
show_trgt_mesh = False
show_frustum = True
show_sampled_points = False
show_sparse_points = False  # deactivated
show_all_points = False  # deactivated
show_grid_points = False
show_corner_points = False
show_zero_point = False
num_encoded_frames = 8


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
    lines.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(edges))])  # Blue lines
    
    return lines
   

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Frustum Sampling Visualization", width=1720, height=980)


# set the point size of all points
render_option = vis.get_render_option()
render_option.point_size = 5.0


for i in range(num_encoded_frames):

    data_folder = f'{base_folder}/frustum_sampling'


    # display frustum sampling
    if show_frustum:
        intrinsics_file = f'{data_folder}/intrinsics_{i}.pt'
        intrinsics_tensor = torch.load(intrinsics_file, map_location=torch.device('cpu'))
        intrinsics_np = intrinsics_tensor.squeeze().numpy()

        pose_file = f'{data_folder}/pose_{i}.pt'
        pose_tensor = torch.load(pose_file, map_location=torch.device('cpu'))
        pose_np = pose_tensor.squeeze().numpy()

        frustum = create_camera_frustum(intrinsics_np, pose_np, near=0.1, far=0.6)
        vis.add_geometry(frustum)

    # all points
    if show_all_points:
        point_cloud_file = f'{data_folder}/all_points_{i}.pt'
        point_cloud_tensor = torch.load(point_cloud_file, map_location=torch.device('cpu'))
        point_cloud_np = point_cloud_tensor.squeeze().numpy()
        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np)
        vis.add_geometry(point_cloud_o3d)

    # sampled points
    if show_sampled_points:
        sampled_point_cloud_file = f'{data_folder}/sampled_points_{i}.pt'
        sampled_point_cloud_tensor = torch.load(sampled_point_cloud_file, map_location=torch.device('cpu'))
        sampled_point_cloud_np = sampled_point_cloud_tensor.squeeze().detach().numpy()
        sampled_point_cloud_o3d = o3d.geometry.PointCloud()
        sampled_point_cloud_o3d.points = o3d.utility.Vector3dVector(sampled_point_cloud_np)
        colors_np = np.ones_like(sampled_point_cloud_np)
        colors_np[:, 0] = 0  # r
        colors_np[:, 1] = 0  # g
        colors_np[:, 2] = 1  # b
        sampled_point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors_np)
        vis.add_geometry(sampled_point_cloud_o3d)
    
        # set point size for sampled points
        vis.update_geometry(sampled_point_cloud_o3d)
        vis.get_render_option().point_size = 5.0



# sparse fps points
data_folder = f'{base_folder}/sparse_points'

if show_sparse_points:
    sparse_point_cloud_file = f'{data_folder}/sparse_points.pt'
    sparse_point_cloud_tensor = torch.load(sparse_point_cloud_file, map_location=torch.device('cpu'))
    sparse_point_cloud_np = sparse_point_cloud_tensor.squeeze().numpy()
    print("pc:", sparse_point_cloud_np.shape)
    sparse_point_cloud_o3d = o3d.geometry.PointCloud()
    sparse_point_cloud_o3d.points = o3d.utility.Vector3dVector(sparse_point_cloud_np)
    sparse_colors_np = np.ones_like(sparse_point_cloud_np)
    sparse_colors_np[:, 0] = 0  # r
    sparse_colors_np[:, 1] = 0  # g
    sparse_colors_np[:, 2] = 1  # b
    sparse_point_cloud_o3d.colors = o3d.utility.Vector3dVector(sparse_colors_np)

    vis.add_geometry(sparse_point_cloud_o3d)



data_folder = f'{base_folder}/test_mesh'

if show_pred_mesh:
    pred_mesh = o3d.io.read_triangle_mesh(f"{data_folder}/test_pred_mesh.ply") # or obj
    pred_mesh.compute_vertex_normals()
    pred_mesh.paint_uniform_color(mesh_color)  # RGB (red)
    vis.add_geometry(pred_mesh)

if show_trgt_mesh:
    trgt_mesh = o3d.io.read_triangle_mesh(f"{data_folder}/test_trgt_mesh.ply") # or obj
    trgt_mesh.compute_vertex_normals()
    trgt_mesh.paint_uniform_color(mesh_color)  # RGB (green)
    vis.add_geometry(trgt_mesh)


def sparsify_points(points, remove_fraction=0.5):
    total_points = points.shape[0]

    # Number of elements to remove
    num_to_remove = int(total_points * remove_fraction)

    # Generate random indices to remove
    indices_to_remove = np.random.choice(total_points, size=num_to_remove, replace=False)

    # Use boolean mask to remove the selected indices
    mask = np.ones(total_points, dtype=bool)
    mask[indices_to_remove] = False

    # create the new array without the 20% of randomly selected elements
    points_subset = points[mask]
    return points_subset


# grid points
if show_grid_points:
    grid_points_file = f'{data_folder}/grid_points.pt'
    grid_points_tensor = torch.load(grid_points_file, map_location=torch.device('cpu'))
    grid_points_np = grid_points_tensor.squeeze().detach().numpy()
    grid_points_np = sparsify_points(grid_points_np, remove_fraction=0.9)
    grid_points_o3d = o3d.geometry.PointCloud()
    grid_points_o3d.points = o3d.utility.Vector3dVector(grid_points_np)

    vis.add_geometry(grid_points_o3d)


if show_corner_points:
    corner_points_file = f'{data_folder}/corner_points.pt'
    corner_points_tensor = torch.load(corner_points_file, map_location=torch.device('cpu'))
    corner_points_np = corner_points_tensor.squeeze().detach().numpy()
    for i in range(len(corner_points_np)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(corner_points_np[i])
        sphere.paint_uniform_color([0, 0, 1])
        vis.add_geometry(sphere)


# mark zero point
if show_zero_point:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    sphere.translate([0.0, 0.0, 0.0])
    sphere.paint_uniform_color([0, 1, 1])  # Blue color
    vis.add_geometry(sphere)


vis.run()
vis.destroy_window()
