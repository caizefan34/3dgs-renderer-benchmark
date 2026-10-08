import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "a0preflight"))
import lib

conics = [(0.5, 0.0, 0.5), (0.8, 0.1, 0.8), (0.3, 0.0, 0.3), (1.0, 0.05, 1.0), (0.6, 0.0, 0.6)]
means2d = [(5.0, 5.0), (6.0, 6.0), (4.0, 5.0), (7.0, 6.0), (5.0, 7.0)]
opacities = [0.9, 0.8, 0.7, 0.6, 0.85]
colors_eval = [(0.2, 0.4, 0.6), (0.5, 0.1, 0.3), (0.1, 0.7, 0.2), (0.9, 0.9, 0.1), (0.4, 0.2, 0.5)]
flatten_ids = [0, 1, 2, 3, 4]
H = W = 16
render_alphas = np.full((1, H, W), 0.5, dtype=np.float32)
last_ids = np.full((1, H, W), 4, dtype=np.int64)
v_image = np.tile(np.array([0.1, 0.2, 0.3], dtype=np.float32), (H, W, 1))

res = lib.tile_bucket_walk(conics, means2d, opacities, colors_eval, 0, 5, flatten_ids,
                           render_alphas, last_ids, 0, 0, W, H, 16, v_image, 0, 2)
h0 = res["accepted"][0]
print("hit0:", {k: (tuple(v) if isinstance(v, tuple) else v) for k, v in h0.items()})

# Hand recompute, step by step, for pixel (0,0), g=4
g = 4
dx = means2d[g][0] - (0 + 0.5)
dy = means2d[g][1] - (0 + 0.5)
print("dx,dy =", dx, dy)
vis, alpha, valid = lib._cpu_eval_gaussian_weight(conics[g], dx, dy, opacities[g])
print("vis, alpha, valid =", vis, alpha, valid)
T_final = 1.0 - float(render_alphas[0, 0, 0])
print("T_final =", T_final)
ra = 1.0 / max(1e-6, 1.0 - alpha)
T = T_final * ra
fac = alpha * T
print("ra, T, fac =", ra, T, fac)
rgbs = colors_eval[g]
vc = [float(v_image[0, 0, k]) for k in range(3)]
print("vc =", vc)
q_color_hand = [fac * vc[k] for k in range(3)]
print("q_color_hand =", q_color_hand)
print("q_color_walk =", [float(h0["q_color"][k]) for k in range(3)])
print("w_color_walk =", float(h0["w_color"]))
