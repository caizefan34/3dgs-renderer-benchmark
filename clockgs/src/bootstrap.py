"""ClockGS bootstrap: wire canonical graphdeco 3DGS onto this env.

- puts .c0-official-reference on sys.path
- shims simple_knn (broken CUDA ext) with torch KD-tree distCUDA2
- provides a pure-torch fused_ssim fallback
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
REF = os.path.join(REPO, ".c0-official-reference")
SHIM = os.path.join(REPO, "clockgs", "src", "shims", "simple_knn_pkg")

for p in (SHIM, REF):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- torch fused_ssim fallback (module: diff_gaussian_rasterization._C lacks fusedssim) ---
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward  # noqa
    _FUSED = True
except Exception:
    _FUSED = False

if not _FUSED:
    import torch
    import torch.nn.functional as F
    from utils.loss_utils import ssim as _torch_ssim

    def fusedssim(C1, C2, img1, img2):
        # img1/img2: [1, C, H, W]; return [1, H, W] map to mirror CUDA fusedssim
        return _torch_ssim(img1, img2, window_size=11, size_average=False)

    def fusedssim_backward(C1, C2, img1, img2, opt_grad):
        # best-effort: route grad through torch ssim; not used by our harness
        raise NotImplementedError("fusedssim_backward not needed for ClockGS oracle")

# expose a fused_ssim(name) like train.py expects
def fused_ssim(img1, img2):
    if _FUSED:
        from diff_gaussian_rasterization import _C
        return _C.fusedssim(0.01 ** 2, 0.03 ** 2, img1, img2).mean()
    return fusedssim(0.01 ** 2, 0.03 ** 2, img1, img2).mean()
