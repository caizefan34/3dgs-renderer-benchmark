#!/usr/bin/env python3
"""Quick smoke test: verify backward call works with one scene."""
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PATH"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:" + os.environ.get("PATH", "")
os.environ["CUDA_HOME"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env"

import torch
import importlib.util
import numpy as np

SOURCE = "/tmp/higs_h3_fwd_1a_source"
CORE_SO = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"

sys.path.insert(0, SOURCE)
spec = importlib.util.spec_from_file_location("gsplat_cuda", CORE_SO)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
sys.modules["gsplat.csrc"] = core

from gsplat.experimental.render.kernels import _backend as _inference_backend
exp = _inference_backend._C
print("Backend loaded:", exp)
print("Has higs_rasterize_backward:", hasattr(exp, "higs_rasterize_backward"))

# Quick test: load room scene and run backward
sys.path.insert(0, "/tmp")
# Import the measurement module functions
import importlib.machinery
loader = importlib.machinery.SourceFileLoader("h5_measure", "/tmp/h5_0r_measure.py")
spec_m = importlib.util.spec_from_loader("h5_measure", loader)
h5m = importlib.util.module_from_spec(spec_m)
loader.exec_module(h5m)

device = torch.device("cuda")
print("\nCapturing forward state for room...")
os.environ["HIGS_PX_RUNTIME"] = "2"
os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
os.environ["H5_FRONTIER"] = "0"
state = h5m.capture_forward_state("room", 2048, device, exp)
print(f"N_vis={state['N_vis']} W={state['width']} H={state['height']}")
print(f"means2d shape: {state['means2d'].shape}")
print(f"render_alphas shape: {state['render_alphas'].shape}")
print(f"last_ids shape: {state['last_ids'].shape}")
print(f"tile_offsets shape: {state['tile_offsets'].shape}")
print(f"backgrounds shape: {state['backgrounds'].shape}")
print(f"visible_ids[:5]: {state['visible_ids'][:5]}")

# Run backward
H, W = state["height"], state["width"]
v_render_colors = torch.randn(1, H, W, 3, device=device, dtype=torch.float32)
v_render_alphas = torch.randn(1, H, W, device=device, dtype=torch.float32)
grads = h5m.make_grad_buffers(state, device)

print("\nRunning backward...")
try:
    outputs = h5m.run_backward(exp, state, v_render_colors, v_render_alphas, grads, device)
    torch.cuda.synchronize()
    print(f"Backward returned {len(outputs)} tensors")
    for i, o in enumerate(outputs):
        print(f"  output[{i}]: shape={o.shape} dtype={o.dtype} max={float(o.abs().max()):.4e}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
