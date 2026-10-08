"""Evaluate original L1/SSIM only on selected cores, with exact five-pixel halo."""

import torch.nn.functional as F


def extract_patches(image_hwc, rows, columns, grid=(9, 15), halo=5):
    height, width, _ = image_hwc.shape
    bh, bw = height//grid[0], width//grid[1]
    assert height % grid[0] == 0 and width % grid[1] == 0
    padded = F.pad(image_hwc.permute(2, 0, 1), (halo, halo, halo, halo))
    patches = padded.unfold(1, bh+2*halo, bh).unfold(2, bw+2*halo, bw)
    return patches[:, rows, columns].permute(1, 0, 2, 3).contiguous()


def patch_weighted_loss(image_hwc, target_patches, rows, columns, grid=(9, 15), inclusion=.25):
    from fused_ssim import FusedSSIMMap
    image = extract_patches(image_hwc, rows, columns, grid)
    ssim = FusedSSIMMap.apply(.01**2, .03**2, image, target_patches, 'same', True, 2)
    pixel = .8*(image-target_patches).abs()+.2*(1-ssim)
    cores = pixel[:, :, 5:-5, 5:-5]
    return cores.mean((1, 2, 3)).sum()/(grid[0]*grid[1]*inclusion)
