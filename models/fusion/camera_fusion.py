import torch
import torch.nn as nn

class CameraFusion(nn.Module):
    def __init__(self, method="concat", out_dim=256):
        super().__init__()
        self.method = method
        self.out_dim = out_dim

    def forward(self, features):
        if isinstance(features, (list, tuple)):
            if self.method == "mean":
                x = torch.stack(features, dim=0).mean(dim=0)
            else:
                x = torch.cat(features, dim=1)
        else:
            x = features
        return x
