import torch

def bev_grid(x_range=(-51.2, 51.2), y_range=(-51.2, 51.2), h=200, w=200, device=None):
    xs = torch.linspace(x_range[0], x_range[1], steps=w, device=device)
    ys = torch.linspace(y_range[0], y_range[1], steps=h, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)

def clamp_bev_indices(ix, iy, h, w):
    ix = torch.clamp(ix, 0, w - 1)
    iy = torch.clamp(iy, 0, h - 1)
    return ix, iy

def boxes3d_to_bev_boxes(boxes3d):
    if boxes3d.numel() == 0:
        return boxes3d.new_zeros((0, 4))
    x, y, l, w = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 3], boxes3d[:, 4]
    x1 = x - l / 2
    y1 = y - w / 2
    x2 = x + l / 2
    y2 = y + w / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)
