# R17 Parameter Input Contract — evidence report

Status: **EXECUTED**

Checkpoint: `C:\Users\36570\3dgs-renderer-benchmark\R17_EXEC_PACK\_dryrun_fixtures\fake_ckpt.pt`
sha256 `b249a8cde4fc86b1b3e37e70d54b8fa42c68c3cc95a5bee991f27524f4711913` — N = 3,000

## 1. What the training stack stores (RAW parameters)

Checkpoint fields come from `gaussian_model.capture()`. Stored values are the
optimizer variables `_scaling`, `_rotation`, `_opacity` — i.e. log-scales,
(approximately unit) quaternion vectors, and logit opacities.

### trainer_capture_fields

`C:\Users\36570\3dgs-renderer-benchmark\baseline\reference_v1\gaussian_model.py` (sha256 `68731e375013a6f2`, line 629)

```
629:     def capture(self) -> dict:
630:         """Save model state for checkpointing."""
631:         return {
632:             "active_sh_degree": self.active_sh_degree,
633:             "xyz": self._xyz.detach().cpu(),
634:             "shs": self._shs.detach().cpu(),
635:             "scaling": self._scaling.detach().cpu(),
636:             "rotation": self._rotation.detach().cpu(),
637:             "opacity": self._opacity.detach().cpu(),
638:             "max_radii2D": self.max_radii2D.detach().cpu(),
639:             "xyz_gradient_accum": self.xyz_gradient_accum.detach().cpu(),
640:             "denom": self.denom.detach().cpu(),
641:             "spatial_lr_scale": self.spatial_lr_scale,
642:             "num_points": self._xyz.shape[0],
643:         }
644: 
645:     def restore(self, state: dict, config: dict):
646:         """Restore from checkpoint."""
647:         self.active_sh_degree = state["active_sh_degree"]
648:         self.spatial_lr_scale = state["spatial_lr_scale"]
649:         self._xyz = nn.Parameter(state["xyz"].cuda().requires_grad_(True))
650:         self._shs = nn.Parameter(state["shs"].cuda().requires_grad_(True))
651:         self._scaling = nn.Parameter(state["scaling"].cuda().requires_grad_(True))
652:         self._rotation = nn.Parameter(state["rotation"].cuda().requires_grad_(True))
653:         self._opacity = nn.Parameter(state["opacity"].cuda().requires_grad_(True))
654:         self.max_radii2D = state["max_radii2D"].cuda()
```


## 2. What the training stack feeds the renderer (ACTIVATED parameters)

`render_with_meta` passes the *getter* outputs: sigmoid(opacity), exp(scaling),
normalize(rotation). The renderer never sees raw logits in the real training loop.

### trainer_getters

`C:\Users\36570\3dgs-renderer-benchmark\baseline\reference_v1\gaussian_model.py` (sha256 `68731e375013a6f2`, line 89)

```
89:     def get_opacity(self):
90:         return torch.sigmoid(self._opacity).squeeze(-1)  # 1D for gsplat
91: 
92:     @property
93:     def get_features(self):
94:         """Return SH coefficients up to active_sh_degree."""
95:         n_active = (self.active_sh_degree + 1) ** 2
96:         return self._shs[:, :n_active, :].contiguous()
97: 
98:     @property
99:     def get_sh_degree(self):
100:         return self.active_sh_degree
101: 
102:     # ---- SH progression (official: oneupSHdegree L145-147) ------------------
103: 
104:     def oneupSHdegree(self):
105:         """Increment active SH degree. Does NOT create new Parameter."""
106:         if self.active_sh_degree < self.max_sh_degree:
107:             self.active_sh_degree += 1
108: 
109:     # ---- Initialization (official: create_from_pcd L149-176) ----------------
110: 
111:     def create_from_pcd(self, pcd_data: dict, spatial_lr_scale: float):
112:         """Initialize from SfM point cloud data.
113: 
114:         Args:
115:             pcd_data: dict with keys 'xyz' [N,3], 'shs' [N,K,3] or None,
116:                       'opacity' [N,1] or None, 'scales' [N,3] or None,
117:                       'rotations' [N,4] or None
118:             spatial_lr_scale: scene extent for LR scheduling
119:         """
```


### trainer_render_activation

`C:\Users\36570\3dgs-renderer-benchmark\baseline\reference_v1\trainer.py` (sha256 `98753fbf35ecc1a2`, line 100)

```
100: def render_with_meta(model, cam, sh_degree, accutile=False):
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

NOT_FOUND / NOT_AVAILABLE


Backward note:

### trainable_backward_note

NOT_FOUND / NOT_AVAILABLE


Frozen entry:

### frozen_entry_docstring

NOT_FOUND / NOT_AVAILABLE


## 4. What the AccuTile CUDA kernel actually computes

### accutile_cuda_t_line

NOT_FOUND / NOT_AVAILABLE


Threshold constants in the frozen tree:

### common_alpha_threshold

`C:\Users\36570\3dgs-renderer-benchmark\R17_EXEC_PACK\_dryrun_fixtures\srcroot\gsplat\cuda\include\Common.h` (sha256 `72c1dbc98da9e8dc`, line 2)

```
1: 
2: #define ALPHA_THRESHOLD (1.f / 255.f)
3: #define GAUSSIAN_EXTEND 1.f
```


### common_gaussian_extend

`C:\Users\36570\3dgs-renderer-benchmark\R17_EXEC_PACK\_dryrun_fixtures\srcroot\gsplat\cuda\include\Common.h` (sha256 `72c1dbc98da9e8dc`, line 3)

```
1: 
2: #define ALPHA_THRESHOLD (1.f / 255.f)
3: #define GAUSSIAN_EXTEND 1.f
```


## 5. Real checkpoint distributions (CPU, whole tensor)

### Opacity (stored raw logits)
- min -9.694605827331543, median -0.06737818941473961, p99 6.925659842491133, max 10.019119262695312
- values outside [0,1]: **2,608 / 3,000**
- logits <= 0 (logf of negative in the t-formula): **1,523**
- logits <= ALPHA_THRESHOLD: 1526
- after sigmoid: median 0.4831618219614029, p99 0.9990186154842376, max 0.999955415725708;
  count <= ALPHA_THRESHOLD: **93**

### Scaling (stored log-scales)
- raw: min -4.701408386230469, median -1.007795512676239, max 2.6481220722198486,
  negative count **7,612**
- exp(raw): min 0.009082476608455181, median 0.36502277851104736, max 14.127483367919922

### Rotation (stored quaternion vectors)
- norm: min 0.11249739676713943, mean 1.8671036958694458, max 4.291562080383301
- fraction with |norm - 1| > 1e-3: **0.999333**

## 6. Branch arithmetic of the CUDA t-formula on the real data

Formula: `t = fminf(1, 2*log(opacity/alpha))` (alpha = 0.00392156862745098).

Raw logits fed as "opacity":
{
  "n": 3000,
  "opacity_le_0_logf_of_negative": 1523,
  "opacity_in_(0,alpha]_t_would_be_negative": 3,
  "opacity_in_(alpha,alpha*e^0.5)_t_in_(0,1)": 1,
  "opacity_ge_alpha*e^0.5_t_saturates_at_1": 1473,
  "frac_with_no_real_threshold_behavior": 0.5086666666666667
}

Sigmoid-activated opacities (the contract):
{
  "n": 3000,
  "opacity_le_0_logf_of_negative": 0,
  "opacity_in_(0,alpha]_t_would_be_negative": 93,
  "opacity_in_(alpha,alpha*e^0.5)_t_in_(0,1)": 49,
  "opacity_ge_alpha*e^0.5_t_saturates_at_1": 2858,
  "frac_with_no_real_threshold_behavior": 0.031
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
