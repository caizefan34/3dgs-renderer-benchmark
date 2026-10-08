import json

# room cam12 tau=2 (from main log + diag_pop)
d = {
    "scene": "room", "camera_id": 12, "tau": 2.0,
    "d1_parity_pass": True,
    "oracle_interaction_ratio": 0.0878,
    "full_cost_fraction_d2plus": 0.4273,
    "full_cost_fraction_d4plus": 0.2646,
    "full_cost_fraction_d8": 0.0928,
    "metrics": {
        "means": {"cosine": 0.9677, "rel_l2": None, "top10_sign": None},
        "scales": {"cosine": 0.9901, "rel_l2": None, "top10_sign": None},
        "quats": {"cosine": 0.9777, "rel_l2": None, "top10_sign": None},
        "opacity": {"cosine": 0.9931, "rel_l2": None, "top10_sign": None},
        "sh0": {"cosine": None, "rel_l2": None, "top10_sign": None},
        "shN": {"cosine": None, "rel_l2": None, "top10_sign": None},
    },
    "step": {"reference_improving": True, "R_step": -0.2218, "P_audit": 0.00235},
    "detail": {
        "n_visible_full": 248500,
        "n_visible_d2": 248712,
        "n_visible_d4": 249153,
        "n_visible_d8": 249992,
        "G_dense": 9.286e-04,
        "G_hyb": -2.060e-04,
    },
}
with open("/tmp/camera_12_tau2_room.json", "w") as f:
    json.dump(d, f, indent=2)
print("wrote /tmp/camera_12_tau2_room.json")
