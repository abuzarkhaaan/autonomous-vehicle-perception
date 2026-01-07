import torch
import torch.nn as nn
from .encoder import BEVFormerEncoder
from .decoder import BEVFormerDecoder

class BEVFormerTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = BEVFormerEncoder(embed_dim=embed_dim, num_layers=num_encoder_layers, dropout=dropout)
        self.decoder = BEVFormerDecoder(embed_dim=embed_dim, num_layers=num_decoder_layers, dropout=dropout)

    def forward(self, multi_cam_features, query_embed, bev_context=None):
        if isinstance(multi_cam_features, (list, tuple)):
            x = torch.cat(multi_cam_features, dim=1)
        else:
            x = multi_cam_features
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.view(b, c, h * w).transpose(1, 2)
        memory = self.encoder(x)
        q = query_embed
        if q.dim() == 2:
            q = q.unsqueeze(0).expand(memory.size(0), -1, -1)
        hs = self.decoder(q, memory)
        return hs
