import torch
import torch.nn as nn

class BEVFormerDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_layers=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8, dropout=dropout, batch_first=True) for _ in range(num_layers)])

    def forward(self, query, memory):
        x = query
        for layer in self.layers:
            x = layer(x, memory)
        return x
