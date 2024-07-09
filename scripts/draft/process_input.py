import cv2
import numpy as np
import torch
import open3d as o3d
from icecream import ic
from utils import read_matrix, camera_to_world
from pathlib import Path


data_path = Path.cwd() / 'data/external/scannet_export'

'''
def get_3d_points(rgb_img, depth_img, depth_intrinsics, trans):
    """
    Parameters:
        rgb_img: color image
        depth_img (4, 4): depth image
        depth_intrinsics (4, 4): depth intrinsics matrix
        trans (4, 4): transformation matrix (camera to world)
    Returns:
        (n, 3): 3D points in world coordinates
        (n, 3): color in rgb for every corresponding point
    """

    fx, fy = depth_intrinsics[0, 0], depth_intrinsics[1, 1]
    cx, cy = depth_intrinsics[0, 2], depth_intrinsics[1, 2]
    
    depth_img = depth_img.astype(np.float32)
    
    # calculate pixel coordinates
    u = np.arange(depth_img.shape[1])
    v = np.arange(depth_img.shape[0])
    u, v = np.meshgrid(u, v)
    
    # calculate 3d coordinates in camera space
    x = (u - cx) * depth_img / fx
    y = (v - cy) * depth_img / fy
    z = depth_img
    
    points_3d = np.stack((x, y, z), axis=-1)
    
    # mask out invalid depth values
    mask = depth_img > 0
    points_3d = points_3d[mask]
    points_3d_world = camera_to_world(points_3d, trans)

    
    # gather corresponding colors
    colors = rgb_img[mask]
    colors = colors[:, ::-1] # reverse the order of color channels (-> cv2.COLOR_RGB2BGR)
    colors = colors/255.0
    
    return points_3d_world, colors
'''


'''
def get_3d_points(rgb_img, depth_img, depth_intrinsics, trans):
    """
    Parameters:
        rgb_img: color image
        depth_img (4, 4): depth image
        depth_intrinsics (4, 4): depth intrinsics matrix
        trans (4, 4): transformation matrix (camera to world)
    Returns:
        (n, 3): 3D points in world coordinates
        (n, 3): color in rgb for every corresponding point
    """

    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    rgb_img_o3d = o3d.geometry.Image(rgb_img)
    depth_img_o3d = o3d.geometry.Image(depth_img)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_img_o3d, depth_img_o3d, convert_rgb_to_intensity=False)

    fx, fy = depth_intrinsics[0, 0], depth_intrinsics[1, 1]
    cx, cy = depth_intrinsics[0, 2], depth_intrinsics[1, 2]

    # get the dimensions of the RGB image
    height, width = rgb_img.shape[:2]

    # create PinholeCameraIntrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)
    
    # transform to world coordinates
    pcd.transform(trans)  # not working!!

    #o3d.visualization.draw_geometries([pcd])

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    return points, colors
'''


def get_3d_points(rgb_img, depth_img, depth_intrinsics, trans):
    """
    Parameters:
        rgb_img: color image
        depth_img (4, 4): depth image
        depth_intrinsics (4, 4): depth intrinsics matrix
        trans (4, 4): transformation matrix (camera to world)
    Returns:
        (n, 3): 3D points in world coordinates
        (n, 3): color in rgb for every corresponding point
    """

    # convert numpy arrays to tensors
    rgb_img_tensor = torch.from_numpy(rgb_img).float()
    depth_img_tensor = torch.from_numpy(depth_img).float()
    depth_intrinsics_tensor = torch.from_numpy(depth_intrinsics).float()
    
    # extract intrinsic parameters
    fx, fy = depth_intrinsics_tensor[0, 0], depth_intrinsics_tensor[1, 1]
    cx, cy = depth_intrinsics_tensor[0, 2], depth_intrinsics_tensor[1, 2]
    
    # calculate image dimensions
    height, width = depth_img.shape
    
    # create meshgrid of pixel coordinates
    u = torch.linspace(0, width - 1, width).repeat(height, 1)
    v = torch.linspace(0, height - 1, height).reshape(-1, 1).repeat(1, width)
    
    # calculate 3d coordinates in camera space
    x = (u - cx) * depth_img_tensor / fx
    y = (v - cy) * depth_img_tensor / fy
    z = depth_img_tensor
    
    points_3d = torch.stack((x, y, z), dim=-1)
    
    # mask out invalid depth values
    mask = depth_img_tensor > 0
    points_3d = points_3d[mask]
    
    # transform to world coordinates
    points_3d_world = camera_to_world(points_3d, trans)

    # gather corresponding colors
    colors = rgb_img_tensor[mask]
    colors = colors[:, [2, 1, 0]] # reverse the order of color channels (-> cv2.COLOR_RGB2BGR)
    colors = colors/255.0
    
    return points_3d_world, colors



def get_point_cloud(scene, frame):

    #color_intr_path = f'{data_path}/{scene}/intrinsic/intrinsic_color.txt'
    depth_intr_path = f'{data_path}/{scene}/intrinsic/intrinsic_depth.txt'
    color_img_path = f'{data_path}/{scene}/color/{frame}.jpg'
    depth_img_path = f'{data_path}/{scene}/depth/{frame}.png'
    pose_path = f'{data_path}/{scene}/pose/{frame}.txt'

    # read intrinsics and pose
    #color_intrinsics = read_matrix(color_intr_path)
    depth_intrinsics = read_matrix(depth_intr_path)
    pose = read_matrix(pose_path)

    # read rgb image
    rgb_img = cv2.imread(color_img_path)


    # read depth image
    depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)  # read as 16-bit image
    depth_shift = 1000.0  # given by scannet https://github.com/ScanNet/ScanNet/tree/master/SensReader/python
    depth_img = depth_img.astype(np.float32) / depth_shift  # to meters

    # resize rgb image to match depth image
    height, width = depth_img.shape
    rgb_img = cv2.resize(rgb_img, (width, height))  # interpolation=cv2.INTER_LINEAR

    # calculate 3d points in world coordinates
    points_3d, colors = get_3d_points(rgb_img, depth_img, depth_intrinsics, pose)
    
    # show images
    depth_image_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow('Depth Image', depth_image_norm)
    cv2.imshow('Color Image', rgb_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points_3d, colors



all_points = None
all_colors = None
scene = 'scene0000_00'

for frame in [1, 600]:

    # get point cloud from frame
    points_3d_world, colors = get_point_cloud(scene, frame)

    # concatenate points
    if all_points is None:
        all_points = points_3d_world
        all_colors = colors
    else:
        all_points = np.concatenate((all_points, points_3d_world), axis=0)
        all_colors = np.concatenate((all_colors, colors), axis=0)
    
    # create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # visualize point cloud
    o3d.visualization.draw_geometries([pcd], window_name='Point Cloud', width=1000, height=1000)
