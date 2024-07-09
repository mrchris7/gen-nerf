import torch
from torch import nn
from src.models.components.pointnet import LocalPoolPointnet
from src.models.components.spatial_encoder import SpatialEncoder


class Encoder(nn.Module):
    """
    Encodes image and point cloud data using three different encoder networks. 
    Feature encodings contain visual, geometric, and context information 
    from an auxiliary model (VLM) and are combined into one feature space.
    """
    
    def __init__(
            self,
            cfg_spatial,
            cfg_pointnet,
            use_auxiliarynet,
            auxiliarynet=None
        ):
        super().__init__()
        
        self.spatial = SpatialEncoder.from_conf(cfg_spatial)
        self.pointnet = LocalPoolPointnet.from_conf(cfg_pointnet)
        self.use_auxiliarynet = use_auxiliarynet
        self.auxiliarynet = auxiliarynet
        
        assert(use_auxiliarynet == (not auxiliarynet is None))


    def forward(self, image, pcd):
        """
        
        """
        feat_spatial = self.spatial(image)  # image (B, C, H, W) -> latent (B, latent_size, H, W)
        feat_pointnet = self.pointnet(pcd)
        
        # concatenate all features
        output = torch.cat((feat_spatial, feat_pointnet), dim=-1)

        # concatenate auxiliary features
        if self.use_auxiliarynet:
            feat_auxiliarynet = self.auxiliarynet(image)
            output = torch.cat((output, feat_auxiliarynet), dim=-1)
        
        return output


    @classmethod
    def from_cfg(cls, cfg, auxiliarynet=None):
        return cls(
            cfg.spatial,
            cfg.pointnet,
            cfg.use_auxiliarynet,
            auxiliarynet,
        )