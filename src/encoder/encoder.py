import torch
from torch import nn
from torch.nn import functional as F
from src.encoder.pointnet import LocalPoolPointnet
from src.encoder.spatial_encoder import SpatialEncoder


class Encoder(nn.Module):
    """
    Encodes image and point cloud data using three different encoder networks. 
    Feature encodings contain visual, geometric, and context (VLM) information 
    and are combined into one feature space.
    """
    
    def __init__(self, cfg, f_teacher):
        super().__init__()
        
        self.resnet = SpatialEncoder.from_conf(cfg)
        self.pointnet = LocalPoolPointnet(cfg)
        self.vlm = f_teacher


    def forward(self, image, pcd):
        feat_resnet = self.resnet(image)  # image (B, C, H, W) -> latent (B, latent_size, H, W)
        feat_pointnet = self.pointnet(pcd)
        feat_vlm = self.vlm(image)

        # concatenate all features
        output = torch.cat((feat_resnet, feat_pointnet, feat_vlm), dim=-1)
        return output


def build_encoder(cfg, teacher):
    return Encoder(
        cfg, teacher
    )