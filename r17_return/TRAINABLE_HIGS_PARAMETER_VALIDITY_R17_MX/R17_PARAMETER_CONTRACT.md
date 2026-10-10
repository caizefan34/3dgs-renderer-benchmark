# R17 Parameter Input Contract — evidence report

Status: **EXECUTED**

Checkpoint: `/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt`
sha256 `ccec425c8da02d423fb66e89450e3c837209f280deea24dbf4a507bc3cd4ea14` — N = 4,413,928

## 1. What the training stack stores (RAW parameters)

Checkpoint fields come from `gaussian_model.capture()`. Stored values are the
optimizer variables `_scaling`, `_rotation`, `_opacity` — i.e. log-scales,
(approximately unit) quaternion vectors, and logit opacities.

### trainer_capture_fields

`/home/liaoyuanjun/3dgs-renderer-benchmark/baseline/reference_v1/gaussian_model.py` (sha256 `b8cb0e251bfb0822`, line 604)

```
604:     def capture(self) -> dict:
605:         """Save model state for checkpointing."""
606:         return {
607:             "active_sh_degree": self.active_sh_degree,
608:             "xyz": self._xyz.detach().cpu(),
609:             "shs": self._shs.detach().cpu(),
610:             "scaling": self._scaling.detach().cpu(),
611:             "rotation": self._rotation.detach().cpu(),
612:             "opacity": self._opacity.detach().cpu(),
613:             "max_radii2D": self.max_radii2D.detach().cpu(),
614:             "xyz_gradient_accum": self.xyz_gradient_accum.detach().cpu(),
615:             "denom": self.denom.detach().cpu(),
616:             "spatial_lr_scale": self.spatial_lr_scale,
617:             "num_points": self._xyz.shape[0],
618:         }
619: 
620:     def restore(self, state: dict, config: dict):
621:         """Restore from checkpoint."""
622:         self.active_sh_degree = state["active_sh_degree"]
623:         self.spatial_lr_scale = state["spatial_lr_scale"]
624:         self._xyz = nn.Parameter(state["xyz"].cuda().requires_grad_(True))
625:         self._shs = nn.Parameter(state["shs"].cuda().requires_grad_(True))
626:         self._scaling = nn.Parameter(state["scaling"].cuda().requires_grad_(True))
627:         self._rotation = nn.Parameter(state["rotation"].cuda().requires_grad_(True))
628:         self._opacity = nn.Parameter(state["opacity"].cuda().requires_grad_(True))
629:         self.max_radii2D = state["max_radii2D"].cuda()
```


## 2. What the training stack feeds the renderer (ACTIVATED parameters)

`render_with_meta` passes the *getter* outputs: sigmoid(opacity), exp(scaling),
normalize(rotation). The renderer never sees raw logits in the real training loop.

### trainer_getters

`/home/liaoyuanjun/3dgs-renderer-benchmark/baseline/reference_v1/gaussian_model.py` (sha256 `b8cb0e251bfb0822`, line 84)

```
84:     def get_opacity(self):
85:         return torch.sigmoid(self._opacity).squeeze(-1)  # 1D for gsplat
86: 
87:     @property
88:     def get_features(self):
89:         """Return SH coefficients up to active_sh_degree."""
90:         n_active = (self.active_sh_degree + 1) ** 2
91:         return self._shs[:, :n_active, :].contiguous()
92: 
93:     @property
94:     def get_sh_degree(self):
95:         return self.active_sh_degree
96: 
97:     # ---- SH progression (official: oneupSHdegree L145-147) ------------------
98: 
99:     def oneupSHdegree(self):
100:         """Increment active SH degree. Does NOT create new Parameter."""
101:         if self.active_sh_degree < self.max_sh_degree:
102:             self.active_sh_degree += 1
103: 
104:     # ---- Initialization (official: create_from_pcd L149-176) ----------------
105: 
106:     def create_from_pcd(self, pcd_data: dict, spatial_lr_scale: float):
107:         """Initialize from SfM point cloud data.
108: 
109:         Args:
110:             pcd_data: dict with keys 'xyz' [N,3], 'shs' [N,K,3] or None,
111:                       'opacity' [N,1] or None, 'scales' [N,3] or None,
112:                       'rotations' [N,4] or None
113:             spatial_lr_scale: scene extent for LR scheduling
114:         """
```


### trainer_render_activation

`/home/liaoyuanjun/3dgs-renderer-benchmark/baseline/reference_v1/trainer.py` (sha256 `ca4f9744daa0edb1`, line 100)

```
100: def render_with_meta(model, cam, sh_degree):
101:     """Render and return (image, meta) with means2d retaining grad."""
102:     data = {
103:         "xyz": model.get_xyz,
104:         "rotations": model.get_rotation,
105:         "scales": model.get_scaling,
106:         "opacity": model.get_opacity,
107:         "shs": model.get_features,
108:     }
109:     r, _, meta = rasterization(
110:         means=data["xyz"],
111:         quats=data["rotations"],
112:         scales=data["scales"],
113:         opacities=data["opacity"],
114:         colors=data["shs"],
115:         viewmats=cam.viewmatrix.unsqueeze(0),
116:         Ks=cam.K.unsqueeze(0),
117:         width=cam.image_width,
118:         height=cam.image_height,
```


## 3. What the HiGS renderer documents as its contract (frozen tree)

tRAINABLE entry docstring (the frozen entry forwards to the same native path):

### trainable_contract_docstring

`/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/experimental/render/functional/gaussian_inference.py` (sha256 `7003ebf5b158b9ea`, line 540)

```
535:         improvement is expected from Stage A.
536: 
537:     Args:
538:         means: Gaussian means. [N, 3] float32 CUDA.
539:         quats: Quaternions (normalized). [N, 4] float32 CUDA.
540:         scales: Scales (activated, positive). [N, 3] float32 CUDA.
541:         opacities: Opacities (activated, in [0,1]). [N] float32 CUDA.
542:         colors: Colors or SH coefficients. If ``sh_degree`` is None, expected
543:             shape [N, D] (post-activation RGB). If ``sh_degree`` is set,
544:             expected shape [N, K, D] (SH coefficients).
545:         out: Optional output buffer (:class:`RenderReturn`). If provided, the
```


Backward note:

### trainable_backward_note

`/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/experimental/render/functional/gaussian_inference.py` (sha256 `7003ebf5b158b9ea`, line 533)

```
531: 
532:     .. note::
533:         Stage A uses the standard gsplat rasterization for the backward pass.
534:         The HiGS path is used only for forward preview. No training speed
535:         improvement is expected from Stage A.
536: 
537:     Args:
```


Frozen entry:

### frozen_entry_docstring

`/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/experimental/render/functional/gaussian_inference.py` (sha256 `7003ebf5b158b9ea`, line 2695)

```
2695: def rasterize_gaussian_higs_frozen(
2696:     means: Tensor,
2697:     quats: Tensor,
2698:     scales: Tensor,
2699:     opacities: Tensor,
2700:     colors: Tensor,
2701:     *,
2702:     sh_degree = None,
2703:     viewmats: Tensor,
2704:     Ks: Tensor,
2705:     width: int,
2706:     height: int,
2707:     tile_size: int = 16,
2708:     near_plane: float = 0.01,
2709:     far_plane: float = 1e10,
2710:     radius_clip: float = 0.0,
2711:     eps2d: float = 0.3,
2712:     background = None,
2713:     render_mode: str = "RGB",
2714:     camera_model: str = "pinhole",
2715:     enable_culling: bool = True,
2716:     use_higs_culling: bool = False,
2717:     return_higs_preview: bool = False,
2718:     freeze_topology: bool = True,
2719:     backward_mode: str = "higs_native",
2720:     sh_compression_mode="none",
2721:     tile_sampling_ratio: float = 1.0,
2722:     sampling_mode: str = "uniform",
2723:     tile_mask=None,
2724:     cull_refresh_interval: int = 1,
2725:     scene=None,
2726: ) -> dict:
2727:     """Differentiable HiGS rendering with per-frame visibility culling (Stage B).
2728: 
2729:     Stage B improves on Stage A by culling invisible Gaussians before
2730:     the differentiable rendering pass. Only visible Gaussians participate
2731:     in the autograd computation.
2732: 
2733:     When ``freeze_topology=True`` (default), the Gaussian count is validated
2734:     across renderer calls. If the count changes, RuntimeError is raised.
2735: 
```


## 4. What the AccuTile CUDA kernel actually computes

### accutile_cuda_t_line

`/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/cuda/csrc/IntersectTile.cu` (sha256 `b2468d28a329e7e5`, line 308)

```
296:         // conic = (a, b, c) = upper triangle of Sigma^{-1}
297:         // Quadratic form: a*dx^2 + 2*b*dx*dy + c*dy^2
298:         const float A = conics[idx * 3];
299:         const float B = conics[idx * 3 + 1];
300:         const float C = conics[idx * 3 + 2];
301: 
302:         // disc = B^2 - A*C = -(det Sigma^{-1})
303:         float disc = B * B - A * C;
304: 
305:         // Opacity-aware isocontour level: alpha = opacity * exp(-0.5 * q) >= ALPHA_THRESHOLD
306:         // => q <= 2 * ln(opacity / ALPHA_THRESHOLD). Cap at GAUSSIAN_EXTEND^2 (same as the gsplat radius budget).
307:         const float opacity = opacities[idx];
308:         float t             = fminf(GAUSSIAN_EXTEND * GAUSSIAN_EXTEND, 2.0f * __logf(opacity / ALPHA_THRESHOLD));
309: 
310:         // SNUGBOX: tight axis-aligned bounding box of the ellipse
311:         float neg_t_over_disc = -t / disc;
312:         float x_extent        = sqrtf(neg_t_over_disc * C);
313:         float y_extent        = sqrtf(neg_t_over_disc * A);
314: 
315:         float2 bbox_min = {mean2d.x - x_extent, mean2d.y - y_extent};
316:         float2 bbox_max = {mean2d.x + x_extent, mean2d.y + y_extent};
317: 
318:         float Bx_over_C    = B * x_extent / C;
319:         float By_over_A    = B * y_extent / A;
320:         float2 bbox_argmin = {mean2d.y + Bx_over_C, mean2d.x + By_over_A};
```


Threshold constants in the frozen tree:

### common_alpha_threshold

`/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/cuda/include/Common.h` (sha256 `31cefccf8159caa8`, line 97)

```
95: constexpr uint32_t kMaxCudaGridDimY = 65535;
96: 
97: #define ALPHA_THRESHOLD         (1.f / 255.f)
98: // GAUSSIAN_EXTEND determines where the gaussian is truncated in standard deviations."
99: #define GAUSSIAN_EXTEND         3.33f
```


### common_gaussian_extend

`/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/cuda/include/Common.h` (sha256 `31cefccf8159caa8`, line 98)

```
96: 
97: #define ALPHA_THRESHOLD         (1.f / 255.f)
98: // GAUSSIAN_EXTEND determines where the gaussian is truncated in standard deviations."
99: #define GAUSSIAN_EXTEND         3.33f
100: // MAX_ALPHA and TRANSMITTANCE_THRESHOLD are chosen so that the equivalent of
```


## 5. Real checkpoint distributions (CPU, whole tensor)

### Opacity (stored raw logits)
- min -6.085175037384033, median -3.81814968585968, p99 6.15418547153469, max 17.32552146911621
- values outside [0,1]: **4,256,075 / 4,413,928**
- logits <= 0 (logf of negative in the t-formula): **4,025,614**
- logits <= ALPHA_THRESHOLD: 4026532
- after sigmoid: median 0.021496174857020378, p99 0.9978799498081207, max 1.0;
  count <= ALPHA_THRESHOLD: **1032**

### Scaling (stored log-scales)
- raw: min -17.302631378173828, median -4.079881429672241, max 1.4663810729980469,
  negative count **13,241,028**
- exp(raw): min 3.058881503648081e-08, median 0.016909470781683922, max 4.333524227142334

### Rotation (stored quaternion vectors)
- norm: min 0.44532811641693115, mean 0.9963197112083435, max 3.201887607574463
- fraction with |norm - 1| > 1e-3: **0.995369**

## 6. Branch arithmetic of the CUDA t-formula on the real data

Formula: `t = fminf(1, 2*log(opacity/alpha))` (alpha = 0.00392156862745098).

Raw logits fed as "opacity":
{
  "n": 4413928,
  "opacity_le_0_logf_of_negative": 4025614,
  "opacity_in_(0,alpha]_t_would_be_negative": 918,
  "opacity_in_(alpha,alpha*e^0.5)_t_in_(0,1)": 598,
  "opacity_ge_alpha*e^0.5_t_saturates_at_1": 386798,
  "frac_with_no_real_threshold_behavior": 0.9122332761204985
}

Sigmoid-activated opacities (the contract):
{
  "n": 4413928,
  "opacity_le_0_logf_of_negative": 0,
  "opacity_in_(0,alpha]_t_would_be_negative": 1032,
  "opacity_in_(alpha,alpha*e^0.5)_t_in_(0,1)": 209008,
  "opacity_ge_alpha*e^0.5_t_saturates_at_1": 4203888,
  "frac_with_no_real_threshold_behavior": 0.0002338053543238585
}

Interpretation rule (deterministic, from the source lines above):
- raw logit <= 0 -> `__logf` of a non-positive number produces NaN; `fminf(1, NaN)`
  returns the non-NaN operand 1.0 -> the *maximum* extent branch is taken.
- 0 < raw logit <= alpha -> `log(opacity/alpha) < 0` -> negative t in the ellipse
  math -> sqrt of negative inside `accutile_ellipse_intersection`.
- activated opacity in (alpha, alpha*e^0.5) -> the only branch with a genuine
  threshold behavior.

## 7. Preliminary contract verdicts (final in R17_FINAL_VERDICT.json)

- OPACITY_INPUT_CONTRACT: **VIOLATED** (R3/R8/R15/R16 fed raw logits)
- SCALE_INPUT_CONTRACT: **VIOLATED** (R3/R8 fed log-scales)
- ROTATION_INPUT_CONTRACT: **VIOLATED** (R8 fed unnormalized; R3 normalized)

These verdicts describe the *harness inputs of R3–R16*, not the checkpoint
(which correctly stores raw optimizer variables) and not the training loop
(which activates before rendering).
