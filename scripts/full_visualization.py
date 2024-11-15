import cv2
import numpy as np
import open3d as o3d
import torch

# copy data from cluster to local:
# > scp -r g101ea13@alex.nhr.fau.de:/home/atuin/g101ea/g101ea13/debug/frustum_sampling .

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


def display_depth(depth, save_file=None, display=True, batch=0):
    """
    Parameters:
        depth (B, H, W): pytorch tensor
        save_file str
        display bool
        batch int
    """
    depth_image_norm = cv2.normalize(depth[batch, :, :].cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if save_file:
        cv2.imwrite(f'{save_file}.png', depth_image_norm)
    if display:
        cv2.imshow("color image", depth_image_norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def display_color(image, save_file=None, display=True, batch=0):
    """
    Parameters:
        color (B, 3, H, W): pytorch tensor
        save_file str
        display bool
        batch int
    """
    color_image = image[batch, :, :, :].cpu().numpy()
    color_image = np.transpose(color_image, (1, 2, 0))  # convert CHW to HWC
    color_image = cv2.normalize(color_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR) # convert to BGR
    if save_file:
        cv2.imwrite(f'{save_file}.png', color_image_bgr)
    if display:
        cv2.imshow("color image", color_image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# CONFIG:
data_folder = '/home/master/Main/iRobMan-Lab2/workspace/gen-nerf/data' 
base_folder = f'{data_folder}'
show_pred_mesh = True
show_trgt_mesh = False
show_trim_mesh = False
show_frustum = False
show_sampled_points = False
show_sparse_points = False
show_depth_frames = True
show_color_frames = True
show_all_points = False
show_grid_points = False
show_corner_points = False
show_zero_point = False
show_gt_mesh = False
num_encoded_frames = 3 # 8
mesh_color = [0.75, 0.75, 0.75]

# LOAD EXPERIMENT:
#base_folder = f'{data_folder}/backups/one_frame_smoothness0.1k10' # frames 1
#base_folder = f'{data_folder}/backups/one_scene_posenc_numfreqs2' # frames 1

# 24.09
#base_folder = f'{data_folder}/backups/one_frame_posenc_freqfactor0.5' # frames 1
#base_folder = f'{data_folder}/backups/one_frame_posenc_onlyspatial' # frames 1
#base_folder = f'{data_folder}/backups/one_frame_posenc_onlypointnet' # frames 1
#base_folder = f'{data_folder}/backups/one_scene_seqs_frames3_baseline' # frames 3
#base_folder = f'{data_folder}/backups/one_scene_seqs_frames10' # frames 10

# 01.10
#base_folder = f'{data_folder}/backups/one_frame/one_frame_spatial_gaussiansmthker41sig10' # frames 1 (*)
#base_folder = f'{data_folder}/backups/one_frame/one_frame_spatial_gaussiansmthker41sig10_nosmthlogtrans' # frames 1

#base_folder = f'{data_folder}/backups/one_frame/smth_log_transf_scale15_beta10_onlypointnet_onlytsdfloss' # frames 1 (*)
#base_folder = f'{data_folder}/backups/one_frame/smth_log_transf_scale15_beta10_onlypointnet' # frames 1


#base_folder = f'{data_folder}/backups/seqs_simple/frames5_seed0' # frames 5
#base_folder = f'{data_folder}/backups/seqs_simple/frames5_seed1' # frames 5
#base_folder = f'{data_folder}/backups/seqs_simple/seqs_seq1_frames7_maxdist2' # frames 7


# 01.10
#base_folder = f'{data_folder}/backups/one_frame/baseline_30epochs' # frames 1
#base_folder = f'{data_folder}/backups/seqs_scene0165/scene0165_seqloc_free' # frames 1
#base_folder = f'{data_folder}/backups/seqs_scene0165/scene0165_seqloc_free_1seq_300epochs' # frames 1

#base_folder = f'{data_folder}/backups/seqs_scene0244_01/evenlyspaced_frames10_1seq' # frames 10


#base_folder = f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_pointnet_500ep_seed0'                 # method
#base_folder = f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_eikonal_500ep_seed0'                  # with L_eik
#base_folder = f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_nologtrans_500ep_seed0'               # without h(x)
#base_folder = f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_noposenc_500ep_seed0'                 # without \gamma(x)
#base_folder = f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_pointnetspatialnoblur_500ep_seed1'    # f_feat & f_vol
#base_folder = f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_spatialnoblur_500ep_seed0'            # f_vol


for i in range(num_encoded_frames):

    data_folder = f'{base_folder}/frustum_sampling'

    # display images
    if show_color_frames:
        color_file = f'{data_folder}/image_{i}.pt'
        color_tensor = torch.load(color_file, map_location=torch.device('cpu'))
        display_color(color_tensor)

    if show_depth_frames:
        depth_file = f'{data_folder}/depth_{i}.pt'
        depth_tensor = torch.load(depth_file, map_location=torch.device('cpu'))
        display_depth(depth_tensor)



vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Frustum Sampling Visualization", width=1720, height=980)

#vis.poll_events()
#vis.update_renderer()

#ctr = vis.get_view_control()
#parameters = o3d.io.read_pinhole_camera_parameters("/home/master/Main/iRobMan-Lab2/workspace/gen-nerf/data/open3d/ablation/ScreenCamera_2024-10-14-11-51-45.json")
#ctr.convert_from_pinhole_camera_parameters(parameters)




# set the point size of all points
render_option = vis.get_render_option()
render_option.point_size = 5.0


for i in range(num_encoded_frames):

    data_folder = f'{base_folder}/frustum_sampling'  # _v2


    # display frustum sampling
    if show_frustum:
        intrinsics_file = f'{data_folder}/intrinsics_{i}.pt'
        intrinsics_tensor = torch.load(intrinsics_file, map_location=torch.device('cpu'))
        intrinsics_np = intrinsics_tensor.squeeze().numpy()

        pose_file = f'{data_folder}/pose_{i}.pt'
        pose_tensor = torch.load(pose_file, map_location=torch.device('cpu'))
        pose_np = pose_tensor.squeeze().numpy()

        frustum = create_camera_frustum(intrinsics_np, pose_np, near=0.1, far=0.6) # far: 4.0 (2.5)     # original: near=0.5, far=4.0
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

if show_trim_mesh:
    trim_mesh = o3d.io.read_triangle_mesh(f"{data_folder}/test_trim_mesh.ply")
    trim_mesh.compute_vertex_normals()
    trim_mesh.paint_uniform_color(mesh_color)  # RGB (red)
    vis.add_geometry(trim_mesh)


def sparsify_points(points, remove_fraction=0.5):
    total_points = points.shape[0]

    # Number of elements to remove
    num_to_remove = int(total_points * remove_fraction)

    # Generate random indices to remove
    indices_to_remove = np.random.choice(total_points, size=num_to_remove, replace=False)

    # Use boolean mask to remove the selected indices
    mask = np.ones(total_points, dtype=bool)
    mask[indices_to_remove] = False

    # Create the new array without the 20% of randomly selected elements
    points_subset = points[mask]
    return points_subset


# grid points
if show_grid_points:
    grid_points_file = f'{data_folder}/grid_points.pt'
    grid_points_tensor = torch.load(grid_points_file, map_location=torch.device('cpu'))
    grid_points_np = grid_points_tensor.squeeze().detach().numpy()
    print("grid_points_np", grid_points_np.shape)
    grid_points_np = sparsify_points(grid_points_np, remove_fraction=0.9)
    print("sparse grid_points_np", grid_points_np.shape)
    grid_points_o3d = o3d.geometry.PointCloud()
    grid_points_o3d.points = o3d.utility.Vector3dVector(grid_points_np)

    vis.add_geometry(grid_points_o3d)


if show_corner_points:
    corner_points_file = f'{data_folder}/corner_points.pt'
    corner_points_tensor = torch.load(corner_points_file, map_location=torch.device('cpu'))
    corner_points_np = corner_points_tensor.squeeze().detach().numpy()
    for i in range(len(corner_points_np)):
        #print(f"corner {i}: {corner_points_np[i]}")
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


if show_gt_mesh:
    data_folder = '/home/master/Main/iRobMan-Lab2/workspace/gen-nerf/dataset/scannet/scans/scene0244_01/mesh_04.ply'
    gt_mesh = o3d.io.read_triangle_mesh(f"{data_folder}") # or obj
    gt_mesh.compute_vertex_normals()
    gt_mesh.paint_uniform_color([0.8, 0.8, 0.8])  # RGB (red)
    translation_vector = [1.4002, 0.3983, 0] # z=-1.4366
    gt_mesh.translate(translation_vector)
    vis.add_geometry(gt_mesh)

'''
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
sphere.translate([6.0, 0.0, 0.0])
sphere.paint_uniform_color([0, 0, 0.5])
vis.add_geometry(sphere)


sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
sphere.translate([0.0, 4.0, 0.0])
sphere.paint_uniform_color([0, 0, 0.3])
vis.add_geometry(sphere)

sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
sphere.translate([6.0, 4.0, 0.0])
sphere.paint_uniform_color([0, 0, 0])
vis.add_geometry(sphere)
'''

vis.run()
vis.destroy_window()
