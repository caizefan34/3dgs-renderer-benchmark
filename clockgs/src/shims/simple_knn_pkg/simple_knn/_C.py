import torch
import numpy as np
from scipy.spatial import cKDTree


def distCUDA2(points):
    pts = points.detach().to("cpu", torch.float32).numpy()
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=4)
    d3 = d[:, 1:4]
    dist2 = (d3 ** 2).mean(axis=1)
    return torch.from_numpy(dist2.astype(np.float32)).to(points.device)
