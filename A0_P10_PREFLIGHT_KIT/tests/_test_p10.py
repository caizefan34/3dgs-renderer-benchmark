import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "a0preflight"))
import lib

# --- synthetic: 1 tile, 5 gaussians, bucket_width=2 (multi-batch) ---
conics = [(0.5, 0.0, 0.5), (0.8, 0.1, 0.8), (0.3, 0.0, 0.3), (1.0, 0.05, 1.0), (0.6, 0.0, 0.6)]
means2d = [(5.0, 5.0), (6.0, 6.0), (4.0, 5.0), (7.0, 6.0), (5.0, 7.0)]
opacities = [0.9, 0.8, 0.7, 0.6, 0.85]
colors_eval = [(0.2, 0.4, 0.6), (0.5, 0.1, 0.3), (0.1, 0.7, 0.2), (0.9, 0.9, 0.1), (0.4, 0.2, 0.5)]
flatten_ids = [0, 1, 2, 3, 4]
H = W = 16
render_alphas = np.full((1, H, W), 0.5, dtype=np.float32)   # T_final = 0.5
last_ids = np.full((1, H, W), 4, dtype=np.int64)
v_image = np.tile(np.array([0.1, 0.2, 0.3], dtype=np.float32), (H, W, 1))

# bucket split: range [0,5), width 2 -> b0:[3,4], b1:[1,2], b2:[0]
batches = lib.tile_batches(0, 5, 2)
assert [b["batch_size"] for b in batches] == [2, 2, 1], "batch split wrong"
print("PASS bucket split:", [(b["batch_index"], b["batch_end"], b["batch_size"]) for b in batches])

res = lib.tile_bucket_walk(conics, means2d, opacities, colors_eval, 0, 5, flatten_ids,
                           render_alphas, last_ids, 0, 0, W, H, 16, v_image, 0, 2)
h0 = res["accepted"][0]
g = h0["g"]
assert g == 4, g
# hand recompute at the ACTUAL hit pixel (h0['I'], h0['J']); T starts at T_final, buffer starts 0
Ih, Jh = h0["I"], h0["J"]
dx = means2d[g][0] - (Jh + 0.5)   # delta.x
dy = means2d[g][1] - (Ih + 0.5)   # delta.y
# EXACT kernel pairing (RasterizeToPixels3DGSDevice.cuh line 47):
#   sigma = 0.5*(conic.x*dx^2 + conic.z*dy^2) + conic.y*dx*dy
# i.e. conic.x -> dx^2, conic.z -> dy^2, conic.y -> dx*dy  (NOT the naive A dx^2 + C dy^2 + B dx dy)
sig = 0.5 * (conics[g][0] * dx ** 2 + conics[g][2] * dy ** 2) + conics[g][1] * dx * dy
alpha = min(0.99, opacities[g] * float(np.exp(-sig)))
T_final = 0.5
ra = 1.0 / max(1e-6, 1.0 - alpha)
T = T_final * ra
fac = alpha * T
rgbs = colors_eval[g]
vc = [float(v_image[Ih, Jh, k]) for k in range(3)]
q_color_hand = [fac * vc[k] for k in range(3)]
v_alpha_hand = sum((rgbs[k] * T - 0.0 * ra) * vc[k] for k in range(3))  # buffer=0 for first hit
print("hit pixel =", (Ih, Jh), "g =", g)
print("q_color walk ", [round(float(h0["q_color"][k]), 7) for k in range(3)])
print("q_color hand ", [round(x, 7) for x in q_color_hand])
print("q_alpha walk ", round(float(h0["q_alpha"]), 7), " hand ", round(float(v_alpha_hand), 7))
assert max(abs(float(h0["q_color"][k]) - q_color_hand[k]) for k in range(3)) < 1e-6, "q_color mismatch"
assert abs(float(h0["q_alpha"]) - v_alpha_hand) < 1e-6, "q_alpha mismatch"
print("PASS: q_color(=fac*v_render_colors) + q_alpha(T from T_final) match hand computation")

qa, qc, wc, mask, cols, cidx = lib.dense_coefficients(res, (0, 0))
assert np.isfinite(qa).all() and np.isfinite(qc).all() and len(cols) <= 128
# mask cell count == accepted hits (exactly one mask cell per accepted (pixel,gaussian));
# q_alpha/q_color/w_color are written only where the mask is set.
assert int(mask.sum()) == res["hits"], (int(mask.sum()), res["hits"])
assert int((wc != 0).sum()) == res["hits"]  # fac > 0 for every accepted hit
# no nonzero coefficient outside the mask
assert int(((~mask) & (qa != 0)).sum()) == 0
assert int(((~mask) & (qc != 0).any(axis=2)).sum()) == 0
print("PASS: dense builder; mask cells == accepted hits =", res["hits"], "; m =", len(cols))

# --- verdict rule ---
rows_fake = [
    {"alpha": {"r99": 3, "r995": 2}, "color_joint": {"r99": 4, "r995": 3}},
    {"alpha": {"r99": 5, "r995": 4}, "color_joint": {"r99": 6, "r995": 5}},
]
v = lib.p10_verdict(rows_fake)
assert v["verdict"] == "A0P_STRONG_PROMOTE", v
rows_fake2 = [
    {"alpha": {"r99": 20, "r995": 10}, "color_joint": {"r99": 3, "r995": 2}},
]
v2 = lib.p10_verdict(rows_fake2)
assert v2["verdict"] == "A0P_DROP", v2
rows_fake3 = [
    {"alpha": {"r99": 10, "r995": 5}, "color_joint": {"r99": 12, "r995": 9}},
]
v3 = lib.p10_verdict(rows_fake3)
assert v3["verdict"] == "A0P_INCONCLUSIVE", v3
print("PASS: verdict rule (STRONG_PROMOTE / DROP / INCONCLUSIVE)")

# --- analyzer on a real bucket npz (true 3-channel q_color) ---
import tempfile, os
tmp = tempfile.mkdtemp()
npz = os.path.join(tmp, "bucket_t0000_b000.npz")
np.savez_compressed(npz, q_alpha=qa, q_color=qc, w_color=wc, accepted_mask=mask,
                    col_gaussians=np.array(cols, dtype=np.int64),
                    accepted_hits=np.array([res["hits"]]), tile_id=np.array([0]))
import oracle_analyze as oa
row = oa.analyze_bucket(npz)
print("analyzer row: m=", row["m"], "m<=128:", row["m_le_bucket_width"],
      "has_w_color:", row["has_w_color"],
      "color_joint energy=", round(row["color_joint"]["energy"], 6))
assert row["m"] <= 128 and row["has_w_color"]
assert row["color_joint"]["energy"] > 0
print("PASS: analyzer consumes true 3-channel q_color + w_color; m <= bucket width")
print("ALL P10 SYNTHETIC TESTS PASS")
