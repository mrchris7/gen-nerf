import cv2
import numpy as np
import open3d as o3d
import pyrender
import trimesh
import torch

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

def get_renderer(mesh, width, height, color=(1.0, 0.0, 0.0), light_pose=None):
    """
    Create a renderer for a given mesh, camera, and its position.

    Parameters:
        mesh (trimesh.Trimesh): The 3D mesh to render.
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
        color (tuple[float, float, float, float]): Color of the the mesh.

    Returns:
        renderer (pyrender.OffscreenRenderer): Pyrender's OffscreenRenderer.
        scene (pyrender.Scene): The rendering scene.
    """
    # create renderer and scene
    renderer = pyrender.OffscreenRenderer(width, height)
    scene = pyrender.Scene()

    # create a material for the mesh
    mesh.visual.vertex_colors = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=color).vertex_colors

    # add mesh to the scene
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node)

    # add light
    light = pyrender.PointLight(color=np.ones(3), intensity=500.0)
    if light_pose is None:
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, 0, 10]
    scene.add(light, pose=light_pose)

    return renderer, scene


def setup_camera(scene, intrinsics, pose, distance_offset):
    """
    Setup the camera parameters of a renderer.

    Parameters:
        scene (pyrender.Scene): The rendering scene.
        intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        pose (np.ndarray): 4x4 camera pose matrix (camera to world).
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
        distance_offset (float): Additional distance to move the camera backward.
    """
    
    # extract camera parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # flip pose for correct camera direction
    pose = pose.clone()
    pose[:3, 2] = -pose[:3, 2]  # flip Z-axis
    pose[:3, 1] = -pose[:3, 1]  # maintain right-handed system

    # optionally: move the camera further behind
    if distance_offset != 0:
        camera_direction = pose[:3, 2]  # z-axis
        pose[:3, 3] += distance_offset * camera_direction

    # create pyrender camera
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    return scene.add(camera, pose=pose.detach().cpu().numpy())


def render(renderer, scene, intrinsics, pose, distance_offset=0):
    """
    Render a color and depth image using the given scene and camera setup.

    Parameters:
        renderer (pyrender.OffscreenRenderer): The renderer used to generate the images.
        scene (pyrender.Scene): The scene that contains the objects to render.
        intrinsics (torch.Tensor): A 3x3 camera intrinsic matrix as a tensor.
        pose (torch.Tensor): A 4x4 camera pose matrix (camera-to-world) as a tensor.
        distance_offset (float): An additional distance to move the camera along the z-axis.

    Returns:
        color_image (np.ndarray): Rendered color image as a numpy array.
        normalized_depth (np.ndarray): Normalized depth image as a numpy array.
    """

    # render
    camera = setup_camera(scene, intrinsics, pose, distance_offset)
    color_image, depth_image = renderer.render(scene)
    scene.remove_node(camera)

    # avoid invalid values
    depth_image = np.asarray(depth_image)    
    if np.any(np.isnan(depth_image)) or np.any(np.isinf(depth_image)):
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=255.0, neginf=0.0)
    
    # normalize the depth image
    max_depth = np.max(depth_image)
    if max_depth > 0:
        normalized_depth = (depth_image / max_depth * 255).astype(np.uint8)
    else:
        normalized_depth = depth_image.astype(np.uint8)

    return color_image, normalized_depth



def compute_camera_pose(mesh, intrinsics, width, height, margin=1.0):
    """
    Computes a pose vector that centers and fully fits the mesh in the camera's field of view.

    Parameters:
        mesh (trimesh.Trimesh): The 3D mesh to render.
        intrinsics (torch.Tensor): 3x3 camera intrinsic matrix.
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
        margin (float): Multiplier to slightly zoom out for padding (default: 1.0).

    Returns:
        torch.Tensor: 4x4 pose matrix (camera-to-world).
    """
    # compute the mesh's axis-aligned bounding box
    bbox = mesh.bounding_box
    bbox_center = torch.tensor(bbox.centroid, dtype=torch.float32)
    bbox_extent = torch.tensor(bbox.extents, dtype=torch.float32)  # dimensions of the bounding box (width, height, depth)

    # compute max dimension of the bounding box
    max_extent = torch.norm(bbox_extent)

    # get focal lengths
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]

    # compute required distance to fit the object
    required_distance_x = (max_extent * margin * fx) / width
    required_distance_y = (max_extent * margin * fy) / height
    required_distance = torch.max(required_distance_x, required_distance_y)

    # set up the camera pose
    camera_position = bbox_center + torch.tensor([0, 0, required_distance], dtype=torch.float32)
    up_direction = torch.tensor([0, 1, 0], dtype=torch.float32)

    # look-at vector (camera-to-target)
    forward_vector = bbox_center - camera_position # order is important!
    forward_vector = forward_vector / torch.norm(forward_vector)

    # compute right vector
    right_vector = torch.cross(up_direction, forward_vector, dim=-1)
    right_vector = right_vector / torch.norm(right_vector)

    # recompute the up vector for orthogonality
    up_vector = torch.cross(forward_vector, right_vector, dim=-1)

    # create 4x4 pose matrix (camera-to-world)
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, 0] = right_vector
    pose[:3, 1] = up_vector
    pose[:3, 2] = forward_vector
    pose[:3, 3] = camera_position

    if pose.is_cuda:
        pose = pose.cpu()

    return pose


# vvvvvvvvvvvvvvvvvv open3d renderer: vvvvvvvvvvvvvvvvvvvvv

def get_renderer_o3d(mesh, width, height, color=(1.0, 0.0, 0.0, 1.0)):
    """
    Create a renderer for a given mesh, camera and its position.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The 3D mesh to render.
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
        color (tuple[float, float, float, float]): Color of the the mesh.

    Returns:
        renderer: Open3D's OffscreenRenderer
    """

    # create renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene

    # setup mesh
    mesh.compute_vertex_normals()

    # create material
    material = o3d.visualization.rendering.MaterialRecord()
    material.base_color = color  # red
    material.shader = "defaultLit"  # lit shader for realistic lighting
    
    scene.add_geometry("mesh", mesh, material)
    
    # setup light
    light_position = np.array([2.0, 3.0, 4.0])  # in world coord
    light_color = np.array([1.0, 1.0, 1.0])  # white
    light_intensity = 1000.0
    light_falloff = 2.0
    cast_shadows = True
    scene.scene.add_point_light("main_light", light_color, light_position, light_intensity, light_falloff, cast_shadows)

    return renderer


def setup_camera_o3d(renderer, intrinsics, pose, width, height, distance_offset=0):
    """
    Setup the camera parameters of a renderer.

    Parameters:
        intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        pose (np.ndarray): 4x4 camera pose matrix (camera to world).
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
        distance_offset (float): Additional distance to move the camera backward.
    """
    # set up intrinsics
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


    # optionally: move the camera further behind
    if distance_offset != 0:
        camera_direction = pose[:3, 2]  # z-axis
        pose[:3, 3] += distance_offset * camera_direction

    # set up extrinsics
    extrinsic = np.linalg.inv(pose)  # convert to world2camera
    renderer.setup_camera(intrinsic, extrinsic)

    
def render_color_image_o3d(renderer, intrinsics, pose, width, height, distance_offset=0):
    """
    Parameters:
        renderer: Open3D's OffscreenRenderer

    Returns:
        image (height, width, 3): color image as numpy array
    """
    renderer = setup_camera_o3d(renderer, intrinsics, pose, width, height, distance_offset)
    rendered_image = np.asarray(renderer.render_to_image())

    return rendered_image


def render_depth_image_o3d(renderer, intrinsics, pose, width, height, distance_offset=0):
    """
    Parameters:
        renderer: Open3D's OffscreenRenderer

    Returns:
        image (height, width): depth image as numpy array
    """
    renderer = setup_camera_o3d(renderer, intrinsics, pose, width, height, distance_offset)
    depth_image = np.asarray(renderer.render_to_depth_image())

    # Normalize the depth map for visualization (optional)
    depth_min, depth_max = np.min(depth_image), np.max(depth_image)
    normalized_depth = ((depth_image - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    
    return normalized_depth
