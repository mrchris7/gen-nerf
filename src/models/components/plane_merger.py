import torch
import torch.nn as nn

class FeaturePlaneMerger(nn.Module):

    def __init__(self, strategy='average', alpha=0.5, c_dim=None):
        super().__init__()
        
        if strategy == 'learn':
            self.conv = nn.Conv2d(c_dim * 2, c_dim, kernel_size=1)
        self.alpha = alpha
        self.strategy = strategy

    def forward(self, plane_1, plane_2):        
        """
        Merge two feature planes a learnable combination of feature planes.
        
        Args:
            plane_1: dict with keys 'xy', 'yz', 'xz', each containing a tensor of shape (B, c_dim, plane_reso, plane_reso).
            plane_2: dict like plane_1.
        
        Returns:
            Merged feature planes as a dict with the same structure as the input.
        """
        if self.strategy == 'average':
            return self.average_merge(plane_1, plane_2)
        elif self.strategy == 'learn':
            return self.learn_merge(plane_1, plane_2)
        else:
            raise NotImplementedError(f"Feature plane merge strategy: {self.strategy}")

    def average_merge(self, plane_1, plane_2):
        """
        merge two feature planes using weighted averaging
        """
        merged_plane = {}
        for key in plane_1.keys():           
            merged_plane[key] = self.alpha * plane_1[key] + (1 - self.alpha) * plane_2[key]
        return merged_plane
    

    def learn_merge(self, plane_1, plane_2):
        """
        merge two feature planes a learnable combination of feature planes
        """
        merged_plane = {}
        for key in plane_1.keys():
            # concatenate along the channel dimension
            combined = torch.cat([plane_1[key], plane_2[key]], dim=1)  # (B, 2*c_dim, H, W)
            # reduce back to original c_dim using 1x1 convolution
            merged_plane[key] = self.conv(combined)
        return merged_plane
    
    @classmethod
    def from_conf(cls, cfg, c_dim=None):
        return cls(
            cfg.strategy,
            cfg.alpha,
            c_dim
        )
