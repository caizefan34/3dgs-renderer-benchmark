"""Frozen per-Gaussian predictors allocated to projected-center blocks.

The allocation need not match true Gaussian support for unbiasedness. Predictive
quality is a separate empirical requirement. All predictor/allocation inputs must
be fixed before sampling and must not receive gradients.
"""

import torch


@torch.no_grad()
def center_blocks(projected, radii, height, width, grid=(9, 15)):
    visible = (radii > 0).all(-1)
    xy = torch.where(visible[..., None], projected, 0.)
    col = (xy[..., 0]/(width/grid[1])).floor().long().clamp(0, grid[1]-1)
    row = (xy[..., 1]/(height/grid[0])).floor().long().clamp(0, grid[0]-1)
    return row*grid[1]+col, visible


@torch.no_grad()
def sampled_prediction(predictor, assignment, labels, group, inclusion=.25):
    weights = (labels.reshape(-1)[assignment] == group).to(predictor.dtype)/inclusion
    return predictor*weights.reshape((-1,)+(1,)*(predictor.ndim-1))
