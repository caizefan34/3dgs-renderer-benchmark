# Numerical Bounds — Candidate C Source Audit

## Purpose
Document every source-enforced numerical bound, clamp, epsilon, and guard in the canonical rendering and training path. These bounds constrain gradient propagation and are essential for bounding analysis.

---

## 1. Opacity Activation

**Source**: `baseline/reference_v1/gaussian_model.py` line 90

```python
@property
def get_opacity(self):
    return torch.sigmoid(self._opacity).squeeze(-1)  # 1D for gsplat
```

| Property | Value | Source | Enforced? |
|----------|-------|--------|-----------|
| Raw opacity → activated | `sigmoid(raw)` | `gaussian_model.py:90` | Always (activation function) |
| Output range | `(0.0, 1.0)` | sigmoid asymptote | Implicit (never exactly 0 or 1) |

---

## 2. Scale Activation

**Source**: `baseline/reference_v1/gaussian_model.py` line 78

```python
@property
def get_scaling(self):
    return torch.exp(self._scaling)
```

| Property | Value | Source | Enforced? |
|----------|-------|--------|-----------|
| Raw scale → activated | `exp(raw)` | `gaussian_model.py:78` | Always |
| Output range | `(0.0, +inf)` | No clamp | No upper bound |
| Minimum raw scale | Not explicitly clamped | — | Not enforced in activation |

gsplat projection applies its own bounds (see below).

---

## 3. Rotation Activation

**Source**: `baseline/reference_v1/gaussian_model.py` line 82

```python
@property
def get_rotation(self):
    return F.normalize(self._rotation, dim=-1)
```

| Property | Value | Source | Enforced? |
|----------|-------|--------|-----------|
| Normalization | L2 normalize | `gaussian_model.py:82` | Always |
| Output | Unit quaternion | — | Numerically |

---

## 4. SH (Color) Representation

**Source**: `baseline/reference_v1/gaussian_model.py` lines 92-98

```python
@property
def get_features(self):
    return self._shs[:, :(self.active_sh_degree + 1)**2]
```

| Property | Value | Source | Enforced? |
|----------|-------|--------|-----------|
| SH coefficients | Direct (no activation) | `gaussian_model.py:52` | No clamp |
| Max SH degree | 3 (max of config) | `gaussian_model.py:46` | Fixed tensor allocation |
| Active SH degree | Progresses from 0 to 3 every 1000 iters | `gaussian_model.py` `oneupSHdegree()` | Gradual |
| SH output range | Unbounded | SH basis evaluation | No clamp |

---

## 5. gsplat Projection Bounds

**Source**: `gsplat/cuda/csrc/fully_fused_projection_fwd.cu` (exact kernel source)

| Property | Value | Source | Enforced? |
|----------|-------|--------|-----------|
| `eps2d` | `0.1` (config default) | `trainer.py:123` | Passed to rasterization call |
| Near plane | Default gsplat: `0.01` | gsplat projection kernel | Implicit in projection |
| Far plane | Default gsplat: `1e10` | gsplat projection kernel | Implicit (very large) |
| Screen bounds | Pixel coordinates clipped to image dims | Raster forward kernel | In tile/pixel mapping |

---

## 6. Raster Forward Bounds

**Source**: `canonical_source/raster_forward/rasterize_to_pixels_fwd.cu`

| Property | Value | Source (line) | Enforced? |
|----------|-------|----------------|-----------|
| Alpha clamp | `min(0.999f, opac * __expf(-sigma))` | Line 146 | Per Gaussian-pixel |
| Alpha minimum | `1.f / 255.f` (skip if below) | Line 147 | Skip threshold |
| Transmittance minimum | `1e-4` (pixel done if T below) | Line 152 | Early termination |
| Background color | Black (0,0,0) by default | Line 181 | When backgrounds=nullptr |
| Tile traversal | Front-to-back by depth sort | CUB radix sort | Always |

---

## 7. Screen-Space Bounds

| Property | Value | Source | Enforced? |
|----------|-------|--------|-----------|
| `radius_clip` | `0.0` (disabled) | `trainer.py:122` | No radius clipping |
| `max_screen_size` | `20` pixels (pruning only) | `configs/reference_v1/room_30k.yaml` | Applied only during densification prune |
| Tile size | `16` | `trainer.py:119` | Fixed |

---

## 8. Optimizer Guards

**Source**: `baseline/reference_v1/gaussian_model.py` training_setup

| Property | Value | Source | Enforced? |
|----------|-------|--------|-----------|
| Adam epsilon | `1e-15` | Config | Standard guard |
| Gradient clipping | `max_norm=1.0` | `torch.nn.utils.clip_grad_norm_` | Applied before optimizer step |

---

## 9. Camera Parameters (Room scene)

**Source**: `configs/reference_v1/room_30k.yaml`

| Property | Value |
|----------|-------|
| Resolution | 1080p (1920×1080) |
| FOV | ~50° (pinhole) |
| Background | black |
| Dataset | mipnerf360/room |

---

## 10. Summary for Gradient Bounding

For the Loss-Aware Pre-Backward Gradient Certificate analysis, the following bounds constrain the backward gradient flow:

1. **Alpha is bounded**: `alpha = min(0.999, opac * exp(-sigma))`, with `opac = sigmoid(raw_opacity) ∈ (0,1)`, so `alpha ∈ (0, 0.999)`.
2. **Transmittance is bounded**: `T ∈ [0, 1]`, early termination at `T < 1e-4`.
3. **Sigma is unconstrained above**: `sigma = 0.5 * (conic.x * dx² + conic.z * dy²) + conic.y * dx * dy` depends on conics, which depend on `exp(scale)` and rotation. Scale is exponentially activated, so sigma can be arbitrarily large (causing near-zero alpha).
4. **Render colors are bounded**: `render_color ∈ [0, 1]` due to clamp in `trainer.py:132`.
5. **No alpha loss**: `v_render_alphas` is always zero in the canonical training loss.

These are **source-enforced** bounds only. Speculative mathematical bounds (e.g., Lipschitz constants of sigmoid/exp) are not documented here.
