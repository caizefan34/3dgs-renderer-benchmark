"""Isolated functional CUDA prototype; fixed parent tile/depth list, RGB only.

Rows contain xy[2], conic[3], color[3], opacity[1]. Paired entries evaluate
two child rows and make pruning/termination decisions on their whole group.
When children move, the caller must supply adequate coverage. Retaining the
parent ordering is a representation choice, not physical child depth sorting.
"""
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_extension = None


def extension():
    global _extension
    if _extension is None:
        _extension = load(
            name="survival_pair_diagnostic_v1",
            sources=[str(Path(__file__).with_name("survival_pair_cuda.cu"))],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=True,
        )
    return _extension


class _Pair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, children, pairs, offsets, ids, bg, width, height, tile):
        for tensor in (base, children, pairs, offsets, ids, bg):
            assert tensor.is_cuda and tensor.is_contiguous()
            assert tensor.device == base.device
        assert base.dtype == children.dtype == bg.dtype == torch.float32
        assert base.ndim == 2 and base.shape[1] == 9
        assert children.ndim == 3 and children.shape[1:] == (2, 9)
        assert pairs.shape == base.shape[:1] and pairs.dtype == ids.dtype == torch.int32
        assert offsets.dtype == torch.int64 and bg.shape == (3,)
        assert not bg.requires_grad, "background VJP is not implemented"
        assert width > 0 and height > 0 and tile > 0
        assert offsets.numel() == ((width + tile - 1) // tile) * ((height + tile - 1) // tile)
        rgb, alpha, last = extension().forward(
            base, children, pairs, offsets, ids, bg, width, height, tile)
        ctx.save_for_backward(base, children, pairs, offsets, ids, bg, alpha, last)
        ctx.size = width, height, tile
        return rgb, alpha

    @staticmethod
    def backward(ctx, v_rgb, v_alpha):
        base, children, pairs, offsets, ids, bg, alpha, last = ctx.saved_tensors
        vb, vk = extension().backward(
            base, children, pairs, offsets, ids, bg, *ctx.size, alpha, last,
            v_rgb.contiguous(), v_alpha.contiguous())
        return vb, vk, None, None, None, None, None, None, None


def rasterize(base, children, pairs, offsets, ids, width, height, background=None, tile=16):
    if background is None:
        background = torch.zeros(3, dtype=base.dtype, device=base.device)
    return _Pair.apply(base, children, pairs, offsets.flatten().contiguous(), ids,
                       background, width, height, tile)
