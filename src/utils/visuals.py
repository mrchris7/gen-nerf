import cv2
import numpy as np


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