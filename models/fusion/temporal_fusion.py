import torch
import torch.nn as nn

class TemporalFusion(nn.Module):
    def __init__(self, method="mean"):
        super().__init__()
        self.method = method

    def forward(self, history):
        if not history:
            return None
        if len(history) == 1:
            return history[0]
        if self.method == "last":
            return history[-1]
        return torch.stack(history, dim=0).mean(dim=0)
