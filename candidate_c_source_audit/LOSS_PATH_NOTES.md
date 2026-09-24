# Loss Path Notes — Candidate C Source Audit

## 1. Loss Function

**Exact source**: `baseline/reference_v1/trainer.py` lines 322-325

```python
L1 = F.l1_loss(image, gt_image)
dssim = ssim_fn(image, gt_image)
loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
```

### Loss function components:

| Component | Source | Function | Weight |
|-----------|--------|----------|--------|
| L1 | `torch.nn.functional.l1_loss` | `F.l1_loss(image, gt_image)` | `1.0 - lambda_dssim` |
| D-SSIM | `trainer.SepSSIM` | `ssim_fn(image, gt_image)` | `lambda_dssim` |

### Weighting constants:

| Constant | Value | Source |
|----------|-------|--------|
| `config.lambda_dssim` | `0.2` | `configs/reference_v1/room_30k.yaml` (parameter `lambda_dssim`) |

---

## 2. L1 Loss

**Function**: `torch.nn.functional.l1_loss`
**Inputs**: `image` (rendered RGB, shape `[H, W, 3]`), `gt_image` (ground truth RGB, shape `[H, W, 3]`)
**Output**: scalar L1 loss
**Derivative**: `dL/d_image = sign(image - gt_image) * (1.0 - lambda_dssim)`

---

## 3. D-SSIM Loss (SepSSIM)

**Class**: `SepSSIM` (defined in `trainer.py` lines 61-87)
**File**: `baseline/reference_v1/trainer.py`

### Forward (lines 74-87):
```python
def __call__(self, pred, target):
    # pred/target: [H, W, 3] → unsqueeze→permute to [1, 3, H, W]
    stacked = torch.cat([pred, target, pred**2, target**2, pred*target], dim=1)
    # 15-channel separable convolution (k_h × k_v = 2D Gaussian kernel)
    b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
    b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
    mu_p, mu_t = b[:, 0:3], b[:, 3:6]
    bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t
    sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
    ssim_map = (2 * mu_pt + self.C1) * (2 * spt + self.C2) \
               / ((mu_p2 + mu_t2 + self.C1) * (sp2 + st2 + self.C2))
    return 1.0 - ssim_map.mean()
```

### Constants:

| Constant | Value | Source |
|----------|-------|--------|
| `self.C1` | `(0.01)**2 = 0.0001` | `window_size=11, sigma=1.5` |
| `self.C2` | `(0.03)**2 = 0.0009` | line 65 |
| `window_size` | `11` | line 64 |
| `sigma` | `1.5` | line 64 |

### Kernel construction (lines 67-71):
```python
coords = torch.arange(11, device=device, dtype=torch.float32) - 5  # 11 // 2
k1d = torch.exp(-(coords**2) / (2 * 1.5**2))
k1d = k1d / k1d.sum()
self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1)
self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2)
```

### D-SSIM backward:
The backward pass goes through the standard PyTorch autograd graph for:
1. `F.conv2d` (separable horizontal then vertical)
2. Arithmetic operations (squares, products, division, mean)
3. The `1.0 - mean` at the end

**Output**: scalar D-SSIM loss, where `D-SSIM = 1.0 - SSIM_value`
**Forward autograd**: SSIM is computed at the per-pixel level with `ssim_map`, then `dssim = 1.0 - ssim_map.mean()`

### Derivative path:
```
dL/loss = 1.0 (scalar)
dL/dssim = lambda_dssim
dL/d_ssim_map = -dL/dssim / (H*W)  (per pixel, from mean backward)
dL/d_pred = dL/d_ssim_map → through SSIM formula → through conv2d
```

---

## 4. Combined Loss Gradient w.r.t. Rendered Output

The total gradient w.r.t. rendered RGB image (`v_render_colors`) is:

```python
v_render_colors = dL/d_image = (1.0 - lambda_dssim) * sign(image - gt_image)  # from L1
                             + lambda_dssim * d(dssim)/d_image               # from D-SSIM
```

Alpha loss is NOT used in the canonical baseline. The loss only uses rendered RGB (`image`), not rendered alpha (`render_alphas`). Therefore:
- `dL/d_render_alphas = [0, H, W, 1]` (zero tensor, no signal from loss)
- The `v_render_alphas` passed to the raster backward kernel is all zeros

**NOTE**: This means the `v_render_alphas` gradient in the backward kernel is always zero for the canonical reference training. Any alpha-gradient computation in the backward kernel is dead work when only RGB loss is used.

---

## 5. Gradient Path Summary

```
loss scalars
  → dL/d_image [H, W, 3]
    → rasterize_to_pixels backward
      → v_render_colors [H, W, 3] (non-zero from loss)
      → v_render_alphas [H, W, 1] (all zeros - no alpha loss)
        → v_colors [N, 3] (from v_render_colors through compositing gradient)
        → v_opacities [N] (from v_render_colors through compositing gradient)
        → v_means2d [N, 2] (from v_render_colors through sigma gradient)
        → v_conics [N, 3] (from v_render_colors through sigma gradient)
```

---

## 6. Source Files Required for External Audit

| File | Role |
|------|------|
| `baseline/reference_v1/trainer.py` lines 61-87 | SepSSIM forward implementation |
| `baseline/reference_v1/trainer.py` lines 322-325 | Loss weighting |
| PyTorch `F.l1_loss` | L1 loss (standard PyTorch) |
| PyTorch `F.conv2d` | SSIM convolution (standard PyTorch) |
