import torch
import torch.nn as nn

class SwinBackbone(nn.Module):
    def __init__(self, out_channels=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(3, out_channels, 4, stride=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)
