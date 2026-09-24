#!/usr/bin/env python3
"""Fix gsplat 1.4.0 API contract in r3_certificate_runner.py:
- spherical_harmonics does NOT accept leading batch dims for coeffs.
- Fix occurrence in _warmup_jsit and main loop.
"""
import io

PATH = "experiments/r3/r3_certificate_runner.py"
with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

# Fix 1: _warmup_jsit -> dirs = xyz - viewmat[:, :3, 3].unsqueeze(1); chang to dirs = xyz[0] - viewmat[0, :3, 3]
old1 = """    dirs = xyz - viewmat[:, :3, 3].unsqueeze(1)
    colors = spherical_harms(3, dirs, shs)"""
new1 = """    dirs = xyz - viewmat[:, :3, 3].unsqueeze(1)
    colors = spherical_harms(3, dirs[0], shs[0])"""

assert src.count(old1) == 1, f"old1 count = {src.count(old1)}"
src = src.replace(old1, new1)

# Fix 2: main loop; use direct flat form
old2 = """        dirs = xyz.unsqueeze(0) - cam.camera_center.unsqueeze(0)
        
        colors_rgb = spherical_harms(
            model.active_degree, dirs, shs.unsqueeze(0)
        )"""
new2 = """        dirs = xyz - cam.camera_center
        
        colors_rgb = spherical_harms(
            model.active_degree, dirs, shs
        )"""
assert src.count(old2) == 1, f"old2 count = {src.count(old2)}"
src = src.replace(old2, new2)

with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Fixed both spherical_harms call sites")
