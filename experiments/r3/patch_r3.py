# -*- coding: utf-8 -*-
"""Apply checkpoint-key normalization fix to r3_certificate_runner.py."""
import io

PATH = "experiments/r3/r3_certificate_runner.py"
with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

old = """    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.restore(ckpt, {"""

new = """    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # ---- Anti-Sabotage Guard: normalize legacy checkpoint keys ----
    state = ckpt.get("model_state", ckpt)
    if "active_sh_degree" not in state and "sh_degree" in state:
        state["active_sh_degree"] = state["sh_degree"]
    if "scaling" not in state and "scales" in state:
        state["scaling"] = state["scales"]
    if "rotation" not in state and "rotations" in state:
        state["rotation"] = state["rotations"]
    if "spatial_lr_scale" not in state:
        state["spatial_lr_scale"] = ckpt.get("metrics", {}).get(
            "spatial_lr_scale", 1.0)
    _n = state["xyz"].shape[0]
    state.setdefault("max_radii2D", torch.zeros(_n, dtype=torch.float32))
    state.setdefault("xyz_gradient_accum", torch.zeros(_n, 1, dtype=torch.float32))
    state.setdefault("denom", torch.zeros(_n, 1, dtype=torch.float32))
    
    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.restore(state, {"""

count = src.count(old)
print("pattern count:", count)
assert count == 1, "expected exactly 1 occurrence"
src = src.replace(old, new)

with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("OK: checkpoint normalization inserted")
