# How to Add a New Renderer

## 1. Implement the Contract

Every new backend must inherit the strict
`RendererAdapter` ABC in `src/adapters/base.py`. The ABC requires checkpoint
loading, rendering, memory reporting, and device metadata. `render_checked`
enforces the public tensor contract; do not duplicate that validation in the
backend wrapper.

Create `src/adapters/my_renderer.py` from this template:

```python
from pathlib import Path
from typing import Any, Dict

import torch

from adapters.base import RendererAdapter


class MyRendererAdapter(RendererAdapter):
    """Thin wrapper around an external CUDA renderer."""

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device=device)
        self.model = None

    def load_checkpoint(self, path: Path) -> Any:
        from my_renderer import load_model

        self.model = load_model(path, device=self.device)
        return self.model

    def render(self, camera_params: Dict) -> Dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("load_checkpoint must be called before render")

        from my_renderer import rasterize

        result = rasterize(
            model=self.model,
            view_matrix=camera_params["view_matrix"],
            projection_matrix=camera_params["projection_matrix"],
            height=camera_params["height"],
            width=camera_params["width"],
        )
        return {
            "rgb": result.rgb.to(dtype=torch.float32).contiguous(),
            "depth": result.depth.to(dtype=torch.float32).contiguous(),
            "alpha": result.alpha.to(dtype=torch.float32).contiguous(),
        }

    def get_memory_usage_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(self.device) / (1024 ** 2)

    def get_device_properties(self) -> Dict:
        properties = torch.cuda.get_device_properties(self.device)
        return {
            "name": properties.name,
            "compute_capability": (
                f"{properties.major}.{properties.minor}"
            ),
            "total_memory_mb": properties.total_memory / (1024 ** 2),
        }
```

Keep the wrapper thin. Checkpoint decoding and static preprocessing belong in
`load_checkpoint`; camera-dependent work belongs in `render` so timing does
not exclude required per-frame work. Never substitute zeros for unsupported
depth or alpha. Add the missing backend capability or document the adapter as
ineligible for Protocol v1.0.

## 2. Register and Test

Export the class from `src/adapters/__init__.py` and add it to the renderer
factory used by the benchmark CLI. Import optional CUDA packages inside
methods so CPU-only test discovery still works.

Add tests to `tests/test_adapters.py` for:

- ABC conformance and construction;
- output keys, float32 dtype, and `[H, W, C]` shape at two resolutions;
- a mocked backend call with the expected camera matrices;
- warmup exclusion and measured-sample ordering;
- quality-gate rejection;
- two-run PSNR drift below 0.01 dB on the synthetic scene.

GPU regression tests must clear references and the CUDA cache between
adapters. A missing optional backend should produce an explicit skip, never a
fabricated pass.

## 3. Validate Quality Before Speed

Render the same camera manifest with the candidate and designated reference.
Run `src/scripts/validate_quality.py` with identical ground-truth images and
record all three metrics. Only after the gate passes should the speed result
be included in a quality-gated leaderboard.

## 4. Document Provenance

Record the upstream repository, exact commit, local patch, build command,
compiler, CUDA toolkit, PyTorch version, driver, and supported GPU
architectures. Include licensing information for code and checkpoints. Submit
the generated JSON artifact rather than transcribing values into a table.
