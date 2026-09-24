import re
with open("/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/rendering.py") as f:
    c = f.read()
old = """                    compute_densify_grad=compute_densify_grad,
                )
            render_colors.append(render_colors_)"""
new = """                    compute_densify_grad=compute_densify_grad,
                    r2_geo_mask=r2_geo_mask,
                    r2_app_mask=r2_app_mask,
                    r2_opacity_mask=r2_opacity_mask,
                )
            render_colors.append(render_colors_)"""
if old in c:
    c = c.replace(old, new, 1)
    with open("/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/rendering.py", "w") as f:
        f.write(c)
    print("Chunked path patched")
else:
    print("Chunked path not found")
