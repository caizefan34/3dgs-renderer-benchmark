"""Anchored control-variate objective. Reference implementation, not sparse CUDA."""
from dataclasses import dataclass
from typing import Callable
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Block:
    y0: int
    y1: int
    x0: int
    x1: int

    def halo(self, height, width, radius=5):
        return Block(max(0, self.y0-radius), min(height, self.y1+radius),
                     max(0, self.x0-radius), min(width, self.x1+radius))


def blocks(height, width, size):
    if min(height, width, size) <= 0:
        raise ValueError("Dimensions and block size must be positive")
    return [Block(y, min(y+size, height), x, min(x+size, width))
            for y in range(0, height, size) for x in range(0, width, size)]


def upsample(image, height, width):
    return F.interpolate(image.permute(2, 0, 1)[None], size=(height, width),
                         mode="bilinear", align_corners=False)[0].permute(1, 2, 0)


@dataclass(frozen=True)
class Anchor:
    fine: torch.Tensor
    coarse: torch.Tensor

    @classmethod
    def capture(cls, fine, coarse):
        # Clones prevent subsequent writes to renderer buffers changing the anchor.
        return cls(fine.detach().clone(), coarse.detach().clone())

    def proxy(self, coarse):
        if coarse.shape != self.coarse.shape:
            raise ValueError("Anchor coarse resolution changed; refresh anchor")
        h, w = self.fine.shape[:2]
        # Detach even manually constructed anchors. All clamps are part of the objective.
        return (self.fine.detach() + upsample(coarse-self.coarse.detach(), h, w)).clamp(0, 1)


def loss_map(pred, target):
    """HWC -> HW: 0.8 RGB-mean L1 + 0.2 (1-SSIM), 11x11, sigma=1.5, zero pad.

    Pure PyTorch reference for the audited SepSSIM formula; fused CUDA parity
    must be checked on the destination baseline before any training comparison.
    """
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("Expected two identically shaped HWC RGB tensors")
    a, b = pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    x = torch.arange(-5, 6, dtype=pred.dtype, device=pred.device)
    k = torch.exp(-x.square()/(2*1.5**2)); k = k/k.sum()
    stacked = torch.cat((a, b, a*a, b*b, a*b), dim=1)
    kh = k.reshape(1, 1, 1, 11).expand(15, 1, 1, 11).contiguous()
    kv = k.reshape(1, 1, 11, 1).expand(15, 1, 11, 1).contiguous()
    stats = F.conv2d(F.conv2d(stacked, kh, padding=(0, 5), groups=15),
                     kv, padding=(5, 0), groups=15)
    ma, mb, aa, bb, ab = stats.split(3, dim=1)
    ssim = ((2*ma*mb+.01**2)*(2*(ab-ma*mb)+.03**2)
            / ((ma*ma+mb*mb+.01**2)*(aa-ma*ma+bb-mb*mb+.03**2)))
    return (.8*(a-b).abs()+.2*(1-ssim))[0].mean(0)


def block_losses(pred, target, partition):
    lm = loss_map(pred, target)
    return torch.stack([lm[b.y0:b.y1, b.x0:b.x1].sum()/lm.numel()
                        for b in partition])


def patch_loss(patch, target_patch, center, support, full_height, full_width):
    """Only centers contribute loss; SSIM halo remains differentiable."""
    if patch.shape[:2] != (support.y1-support.y0, support.x1-support.x0):
        raise ValueError("Patch and support dimensions differ")
    if center.halo(full_height, full_width) != support:
        raise ValueError("Support must be the exact 5-pixel clipped halo")
    lm = loss_map(patch, target_patch)
    return lm[center.y0-support.y0:center.y1-support.y0,
              center.x0-support.x0:center.x1-support.x0].sum()/(full_height*full_width)


def probabilities(pi, n, reference):
    p = torch.as_tensor(pi, dtype=reference.dtype, device=reference.device).detach()
    if p.ndim == 0:
        p = p.expand(n)
    if p.shape != (n,) or not torch.isfinite(p).all() or not ((p > 0) & (p <= 1)).all():
        raise ValueError("Every block needs a finite inclusion probability in (0, 1]")
    return p


def draw(pi, n, reference, generator=None):
    p = probabilities(pi, n, reference)
    return torch.rand(p.shape, device=p.device, generator=generator) < p, p


def corrected_loss(fine_parts, proxy_parts, selected, pi):
    p = probabilities(pi, proxy_parts.numel(), proxy_parts)
    z = torch.as_tensor(selected, device=p.device)
    if z.shape != p.shape or z.dtype != torch.bool or fine_parts.shape != p.shape:
        raise ValueError("Expected Boolean selection and one loss contribution per block")
    return proxy_parts.sum() + ((z.to(p.dtype)/p)*(fine_parts-proxy_parts)).sum()


def reference_step(render_fine: Callable, coarse, target, anchor, partition,
                   pi, generator=None):
    """No optimizer/ADC changes. Fine renderer is a FULL-IMAGE diagnostic fallback.

    Returns an objective for a normal dense .backward()/Adam step. A future
    sparse renderer must produce exact full-camera halo patches for this API.
    """
    proxy = anchor.proxy(coarse)
    pparts = block_losses(proxy, target, partition)
    selected, p = draw(pi, len(partition), pparts, generator)
    h, w = target.shape[:2]
    loss = pparts.sum()
    # No forced sample when empty: forced resampling would change inclusion probabilities.
    if selected.any():
        fine = render_fine()
        for i in selected.nonzero().flatten().tolist():
            b = partition[i]; s = b.halo(h, w)
            f = patch_loss(fine[s.y0:s.y1, s.x0:s.x1], target[s.y0:s.y1, s.x0:s.x1],
                           b, s, h, w)
            loss = loss + (f-pparts[i])/p[i]
    return loss, {"selected": selected, "pi": p, "fine_path": "full_image_reference"}
