"""Unbiased two-resolution loss estimator; control-variate algebra is standard."""

import torch


def block_means(loss_map, rows, columns):
    """Equal-area image partition. Map is H x W after channel averaging."""
    height, width = loss_map.shape
    assert height % rows == 0 and width % columns == 0
    return loss_map.reshape(rows, height//rows, columns, width//columns).mean((1, 3))


def corrected_loss(fine_blocks, coarse_blocks, inclusion_weights, beta=1.):
    """Weights are indicator/inclusion probability, independent of current draws.

    Beta must be fixed before drawing the sampled blocks. The coarse proxy may
    have any differentiable definition, but sampled and full proxy must match.
    """
    assert fine_blocks.shape == coarse_blocks.shape == inclusion_weights.shape
    assert not inclusion_weights.requires_grad
    return (inclusion_weights*fine_blocks).mean() + beta*((1-inclusion_weights)*coarse_blocks).mean()


def fused_loss_map(image_hwc, target_hwc):
    from fused_ssim import FusedSSIMMap
    image = image_hwc.permute(2, 0, 1).unsqueeze(0).contiguous()
    target = target_hwc.permute(2, 0, 1).unsqueeze(0).contiguous()
    ssim = FusedSSIMMap.apply(.01**2, .03**2, image, target, 'same', True, 2)
    return (.8*(image-target).abs()+.2*(1-ssim)).mean(dim=1)[0]
