import torch

def make_homogeneous(points):
    if points.numel() == 0:
        return points.new_zeros((0, 4))
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    return torch.cat([points[:, :3], ones], dim=-1)

def transform_points(points, T):
    if points.numel() == 0:
        return points
    ph = make_homogeneous(points)
    out = (ph @ T.t())[:, :3]
    return out

def project_points(points_3d, K):
    if points_3d.numel() == 0:
        return points_3d.new_zeros((0, 2))
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = torch.clamp(points_3d[:, 2], min=1e-6)
    u = (K[0, 0] * x + K[0, 2] * z) / z
    v = (K[1, 1] * y + K[1, 2] * z) / z
    return torch.stack([u, v], dim=-1)
