import torch
import torch.nn as nn
from .encoder import BEVFormerEncoder
from .decoder import BEVFormerDecoder
from .transformer import BEVFormerTransformer

class BEVFormer(nn.Module):
    def __init__(self, embed_dim=256, num_queries=900, num_classes=10, num_decoder_layers=6, num_encoder_layers=6):
        super().__init__()
        self.transformer = BEVFormerTransformer(embed_dim=embed_dim, num_decoder_layers=num_decoder_layers, num_encoder_layers=num_encoder_layers)
        self.class_head = nn.Linear(embed_dim, num_classes)
        self.box_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 9))
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.query_embed = nn.Embedding(num_queries, embed_dim)

    def forward(self, multi_cam_features, bev_context=None):
        q = self.query_embed.weight.unsqueeze(0)
        hs = self.transformer(multi_cam_features, q, bev_context=bev_context)
        logits = self.class_head(hs)
        boxes = self.box_head(hs)
        return {"logits": logits, "boxes": boxes}
