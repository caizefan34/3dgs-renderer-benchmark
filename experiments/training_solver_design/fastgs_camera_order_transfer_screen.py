"""Check frozen room@4100 camera order against a later full-resolution checkpoint."""

import json
from pathlib import Path

import numpy as np

from fastgs_camera_order_sketch_screen import greedy_order, max_prefix_norm, summarize


ROOT = Path(__file__).parent


def load_centered(path):
    with np.load(path) as source:
        gram = source["gram"].astype(np.float64)
        names = source["camera_names"].tolist()
    centered = gram - gram.mean(axis=0)[None, :] - gram.mean(axis=1)[:, None] + gram.mean()
    return (centered + centered.T) / 2, names


def main():
    first, names_first = load_centered(ROOT / "fastgs_camera_xyz_gram.npz")
    order_first = greedy_order(first)
    rng = np.random.default_rng(20260927)
    result = {
        "camera_count": len(first),
        "room_4100_oracle_on_4100": max_prefix_norm(first, order_first),
        "later_checkpoints": {},
        "scope": "Transfer of an offline oracle order across frozen room xyz-gradient Gram matrices. Not online order training, Adam speed, unseen-view quality, or a new-scene validation."
    }
    for iteration in (5000, 10000, 30000):
        later, names_later = load_centered(ROOT / f"fastgs_camera_{iteration}_xyz_gram.npz")
        if len(set(names_first)) != len(names_first) or set(names_first) != set(names_later):
            raise ValueError(f"Camera names differ or are duplicated at {iteration}")
        index_later = {name: i for i, name in enumerate(names_later)}
        transferred = [index_later[names_first[i]] for i in order_first]
        order_later = greedy_order(later)
        random_scores = [max_prefix_norm(later, rng.permutation(len(later))) for _ in range(100)]
        transfer_score = max_prefix_norm(later, transferred)
        oracle_score = max_prefix_norm(later, order_later)
        row = {"transferred_max_prefix_norm": transfer_score,
               "later_oracle_max_prefix_norm": oracle_score,
               "random_max_prefix_norm": summarize(random_scores),
               "transfer_ratio_to_random_median": transfer_score / np.median(random_scores),
               "transfer_ratio_to_later_oracle": transfer_score / oracle_score}
        result["later_checkpoints"][str(iteration)] = row
    output = ROOT / "fastgs_camera_order_transfer_result.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
