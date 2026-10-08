"""Screen inexpensive pose-only camera orders against frozen FastGS gradients."""

import json
from pathlib import Path

import numpy as np

from fastgs_camera_order_sketch_screen import greedy_order, max_prefix_norm


ROOT = Path(__file__).parent
ITERATIONS = (4100, 5000, 10000, 30000)


def load_gram(iteration):
    label = "" if iteration == 4100 else f"_{iteration}"
    with np.load(ROOT / f"fastgs_camera{label}_xyz_gram.npz") as source:
        gram = source["gram"].astype(np.float64)
        names = source["camera_names"].tolist()
    centered = gram - gram.mean(axis=0)[None, :] - gram.mean(axis=1)[:, None] + gram.mean()
    return (centered + centered.T) / 2, names


def pairwise_squared(values):
    norms = np.sum(values**2, axis=1)
    return np.maximum(norms[:, None] + norms[None, :] - 2 * values @ values.T, 0)


def main():
    grams = {}
    for iteration in ITERATIONS:
        grams[iteration], names = load_gram(iteration)
        if iteration == 4100:
            training_names = names
        elif names != training_names:
            raise ValueError(f"Camera order changed at {iteration}")
    metadata = json.loads((ROOT / "fastgs_room_cameras.json").read_text(encoding="utf-8"))
    cameras = {row["img_name"]: row for row in metadata}
    position = np.asarray([cameras[name]["position"] for name in training_names])
    rotation = np.asarray([cameras[name]["rotation"] for name in training_names]).reshape(len(names), -1)
    position_distance = pairwise_squared(position)
    rotation_distance = pairwise_squared(rotation)
    upper = np.triu_indices(len(names), 1)
    position_distance /= np.median(position_distance[upper])
    rotation_distance /= np.median(rotation_distance[upper])
    rows = []
    for position_weight, rotation_weight in ((1, 0), (0, 1), (1, .25), (1, 1), (.25, 1)):
        for bandwidth in (.5, 1, 2, 4):
            distance = position_weight * position_distance + rotation_weight * rotation_distance
            kernel = np.exp(-distance / (2 * bandwidth**2))
            centered = kernel - kernel.mean(axis=0)[None, :] - kernel.mean(axis=1)[:, None] + kernel.mean()
            order = greedy_order(centered)
            scores = {str(iteration): max_prefix_norm(grams[iteration], order) for iteration in ITERATIONS}
            rows.append({"position_weight": position_weight, "rotation_weight": rotation_weight,
                         "bandwidth": bandwidth, "true_gradient_max_prefix_norm": scores,
                         "camera_order": [training_names[i] for i in order]})
    rows.sort(key=lambda row: row["true_gradient_max_prefix_norm"]["4100"])
    result = {"camera_count": len(names), "ranked_by": "room@4100 true gradient proxy; development-only model selection",
              "candidates": rows, "scope": "Pose-only RBF camera ordering evaluated on frozen room gradient Gram matrices. No trainer, full wall cost, or novel-view quality result; camera metadata includes official test cameras but only training names were used."}
    output = ROOT / "fastgs_camera_pose_order_result.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in rows[:5]:
        print({key: value for key, value in row.items() if key != "camera_order"})


if __name__ == "__main__":
    main()
