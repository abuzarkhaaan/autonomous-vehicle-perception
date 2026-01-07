import torch
import torch.nn as nn

class LidarFusion(nn.Module):
    def __init__(self, enabled=False):
        super().__init__()
        self.enabled = enabled

    def forward(self, bev_feat, lidar_feat=None):
        if not self.enabled or lidar_feat is None:
            return bev_feat
        if bev_feat.shape[-2:] == lidar_feat.shape[-2:]:
            return bev_feat + lidar_feat
        return bev_feat
