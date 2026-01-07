import torch

def box_iou_2d(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    xa1 = torch.maximum(x11[:, None], x21[None, :])
    ya1 = torch.maximum(y11[:, None], y21[None, :])
    xa2 = torch.minimum(x12[:, None], x22[None, :])
    ya2 = torch.minimum(y12[:, None], y22[None, :])
    inter_w = torch.clamp(xa2 - xa1, min=0)
    inter_h = torch.clamp(ya2 - ya1, min=0)
    inter = inter_w * inter_h
    area1 = torch.clamp(x12 - x11, min=0) * torch.clamp(y12 - y11, min=0)
    area2 = torch.clamp(x22 - x21, min=0) * torch.clamp(y22 - y21, min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / torch.clamp(union, min=1e-9)

def accuracy_top1(logits, targets):
    if logits.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()

def average_precision_stub():
    return {"mAP": None, "NDS": None}
