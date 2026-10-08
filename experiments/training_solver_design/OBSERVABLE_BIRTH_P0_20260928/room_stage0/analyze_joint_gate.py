"""Count families passing the prespecified per-view 1% geometry screen."""

import json
from pathlib import Path

import numpy as np


root = Path(__file__).parent
raw = np.load(root / "stage0_geometry_raw.npz")
G = raw["G"].astype(np.float64)
active = raw["active"]
children = raw["historical_native_child_counts"]
A = np.arange(0, 12, 2)
B = np.arange(1, 12, 2)
covered = (active[A].sum(axis=0) >= 3) & (active[B].sum(axis=0) >= 2)
idx = np.flatnonzero(covered)
summed = G[A][:, idx].sum(axis=0)
_, q = np.linalg.eigh(summed)
weak = q[:, :, 0]
response = np.einsum("ni,vnij,nj->vn", weak, G[:, idx], weak)
trace = np.trace(G[:, idx], axis1=2, axis2=3)
frac = response / np.maximum(trace, 1e-30)
pass_each = (~active[:, idx]) | ((trace > 0) & (frac <= 0.01))
passed = pass_each.all(axis=0)
print(json.dumps({
    "covered_parents": int(len(idx)),
    "covered_native_children": int(children[idx].sum()),
    "all_active_12_views_at_most_1pct_parents": int(passed.sum()),
    "all_active_12_views_at_most_1pct_native_children": int(children[idx[passed]].sum()),
    "fraction_of_all_three_axis_native_children": float(children[idx[passed]].sum() / children.sum()),
}, indent=2))
