"""
Reference V1 instrumentation — C49/C50/C53 data collection during the SAME 30K run.

Collects:
  C49: gradient concentration (top-K mass, Gini)
  C50: temporal gradient predictability (lag-1 Pearson/Spearman, top-K persistence)
  C53: workload statistics (tiles_per_gauss, screen radius, visibility, persistence)

Also maintains stable Gaussian identity/lineage for tracking across topology changes.
"""

import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class GaussianIdentityTracker:
    """Maintain stable logical IDs for audit instrumentation.

    NOT a training algorithm — purely for tracking lineage.
    """

    def __init__(self, n_initial: int, device: str = "cuda"):
        self.device = device
        self.next_id = 0
        self.max_id = 0

        # Per-ID metadata
        self.birth_iter = {}  # id -> iteration
        self.birth_type = {}  # id -> "initial" | "clone" | "split_child"
        self.parent_id = {}   # id -> parent_id or -1
        self.death_iter = {}  # id -> iteration (if pruned)

        # Current mapping: index_in_tensor -> stable_id
        self.index_to_id = torch.arange(n_initial, dtype=torch.long, device=device)
        self.next_id = n_initial
        self.max_id = n_initial

        for i in range(n_initial):
            self.birth_iter[i] = 0
            self.birth_type[i] = "initial"
            self.parent_id[i] = -1

    def extend(self, n_new: int, iter_idx: int, birth_type: str = "unknown",
               parent_ids: Optional[torch.Tensor] = None):
        """Add new IDs for newly created Gaussians."""
        new_ids = torch.arange(self.next_id, self.next_id + n_new,
                               dtype=torch.long, device=self.device)
        self.index_to_id = torch.cat([self.index_to_id, new_ids])

        for i in range(n_new):
            gid = self.next_id + i
            self.birth_iter[gid] = iter_idx
            self.birth_type[gid] = birth_type
            if parent_ids is not None and i < len(parent_ids):
                self.parent_id[gid] = int(parent_ids[i])
            else:
                self.parent_id[gid] = -1

        self.next_id += n_new
        self.max_id = self.next_id

    def filter(self, keep_mask: torch.Tensor, iter_idx: int):
        """Remove IDs for pruned Gaussians."""
        pruned_ids = self.index_to_id[~keep_mask]
        for pid in pruned_ids.cpu().tolist():
            self.death_iter[pid] = iter_idx
        self.index_to_id = self.index_to_id[keep_mask]

    def get_ids(self) -> torch.Tensor:
        return self.index_to_id

    def get_age(self, iter_idx: int) -> torch.Tensor:
        """Return age (iterations since birth) for all current Gaussians."""
        ids = self.index_to_id.cpu().tolist()
        ages = torch.tensor([iter_idx - self.birth_iter.get(gid, 0) for gid in ids],
                           dtype=torch.float32, device=self.device)
        return ages

    def summary(self) -> dict:
        alive = self.index_to_id.cpu().tolist()
        n_initial = sum(1 for gid in alive if self.birth_type.get(gid) == "initial")
        n_clone = sum(1 for gid in alive if self.birth_type.get(gid) == "clone")
        n_split = sum(1 for gid in alive if self.birth_type.get(gid) == "split_child")
        n_dead = len(self.death_iter)
        return {
            "total_ids_assigned": self.max_id,
            "currently_alive": len(alive),
            "alive_initial": n_initial,
            "alive_clone": n_clone,
            "alive_split_child": n_split,
            "total_deaths": n_dead,
        }


class C49GradientConcentration:
    """C49 recalibration: gradient concentration statistics."""

    def __init__(self):
        self.checkpoints = {}

    def record(self, iter_idx: int, grad_norms: torch.Tensor):
        """Record gradient concentration at a checkpoint.

        Args:
            grad_norms: [N] per-Gaussian gradient norm (view-space mean2D)
        """
        if iter_idx not in self.checkpoints:
            self.checkpoints[iter_idx] = {}

        grad_np = grad_norms.detach().cpu().numpy()
        n = len(grad_np)
        if n == 0:
            return

        sorted_grad = np.sort(grad_np)[::-1]  # descending
        total = sorted_grad.sum()
        if total <= 0:
            return

        # Top-K mass
        for k_pct in [1, 5, 10, 20, 32, 50]:
            k = max(1, int(n * k_pct / 100))
            mass = sorted_grad[:k].sum() / total
            self.checkpoints[iter_idx][f"top{k_pct}_mass"] = float(mass)

        # Gini coefficient
        sorted_asc = np.sort(grad_np)
        cumsum = np.cumsum(sorted_asc)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_asc) /
                (n * cumsum[-1]) - (n + 1) / n) if cumsum[-1] > 0 else 0
        self.checkpoints[iter_idx]["gini"] = float(gini)
        self.checkpoints[iter_idx]["n_gaussians"] = n
        self.checkpoints[iter_idx]["mean_grad"] = float(grad_np.mean())
        self.checkpoints[iter_idx]["median_grad"] = float(np.median(grad_np))
        self.checkpoints[iter_idx]["max_grad"] = float(grad_np.max())

    def to_dict(self) -> dict:
        return {"checkpoints": {str(k): v for k, v in self.checkpoints.items()}}


class C50TemporalPredictability:
    """C50 recalibration: temporal gradient predictability."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.prev_grad = None  # [N] previous iteration gradient norms
        self.prev_ids = None   # [N] previous iteration IDs
        self.checkpoints = {}

    def update(self, iter_idx: int, grad_norms: torch.Tensor, ids: torch.Tensor):
        """Record lag-1 temporal correlation.

        Compares g_i(t-1) → g_i(t) for Gaussians alive at both t-1 and t.
        """
        if self.prev_grad is not None and self.prev_ids is not None:
            # Ensure prev_grad and prev_ids have same size
            min_prev_len = min(len(self.prev_grad), len(self.prev_ids))
            prev_grad_safe = self.prev_grad[:min_prev_len]
            prev_ids_safe = self.prev_ids[:min_prev_len]

            # Match IDs between t-1 and t
            prev_ids_set = prev_ids_safe.cpu().numpy()
            curr_ids_set = ids.cpu().numpy()

            # Build lookup: id -> index in current
            curr_lookup = {}
            for i, gid in enumerate(curr_ids_set):
                curr_lookup[int(gid)] = i

            # Find common IDs
            prev_vals = []
            curr_vals = []
            for i, gid in enumerate(prev_ids_set):
                gid_int = int(gid)
                if gid_int in curr_lookup:
                    prev_vals.append(float(prev_grad_safe[i]))
                    curr_vals.append(float(grad_norms[curr_lookup[gid_int]]))

            if len(prev_vals) >= 10:
                prev_arr = np.array(prev_vals)
                curr_arr = np.array(curr_vals)

                # Pearson
                if np.std(prev_arr) > 1e-10 and np.std(curr_arr) > 1e-10:
                    pearson = float(np.corrcoef(prev_arr, curr_arr)[0, 1])
                else:
                    pearson = 0.0

                # Spearman (rank correlation)
                prev_rank = np.argsort(np.argsort(prev_arr))
                curr_rank = np.argsort(np.argsort(curr_arr))
                if np.std(prev_rank) > 1e-10 and np.std(curr_rank) > 1e-10:
                    spearman = float(np.corrcoef(prev_rank, curr_rank)[0, 1])
                else:
                    spearman = 0.0

                # Top-K persistence
                n = len(prev_arr)
                for k_pct in [1, 5, 10, 20, 32, 50]:
                    k = max(1, int(n * k_pct / 100))
                    top_prev_idx = np.argsort(prev_arr)[::-1][:k]
                    top_curr_idx = np.argsort(curr_arr)[::-1][:k]
                    jaccard = len(set(top_prev_idx) & set(top_curr_idx)) / k
                    self.checkpoints.setdefault(iter_idx, {})[f"top{k_pct}_jaccard"] = float(jaccard)

                self.checkpoints.setdefault(iter_idx, {}).update({
                    "n_matched": len(prev_vals),
                    "pearson": pearson,
                    "spearman": spearman,
                })

        self.prev_grad = grad_norms.detach().clone()
        self.prev_ids = ids.detach().clone()

    def to_dict(self) -> dict:
        return {"checkpoints": {str(k): v for k, v in self.checkpoints.items()}}


class C53WorkloadStatistics:
    """C53 recalibration: workload distribution and persistence."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.checkpoints = {}
        # Per-iteration trace for sampled cohort
        self.cohort_size = 100_000
        self.cohort_ids = None  # Fixed cohort of stable IDs to track
        self.cohort_trace = {}  # iter -> {id: {tiles, visible, radius}}

    def init_cohort(self, ids: torch.Tensor):
        """Initialize fixed cohort of IDs to track."""
        n = min(self.cohort_size, len(ids))
        self.cohort_ids = ids[:n].detach().clone()

    def record_checkpoint(self, iter_idx: int, tiles_per_gauss: torch.Tensor,
                          radii: torch.Tensor, ids: torch.Tensor,
                          scales: torch.Tensor):
        """Record workload statistics at a checkpoint."""
        if iter_idx not in self.checkpoints:
            self.checkpoints[iter_idx] = {}

        tiles_np = tiles_per_gauss.detach().cpu().numpy()
        visible = (radii > 0).any(dim=-1).cpu().numpy() if radii.dim() > 1 else (radii > 0).cpu().numpy()
        scale_norm = scales.detach().norm(dim=-1).cpu().numpy()

        n = len(tiles_np)
        self.checkpoints[iter_idx]["n_gaussians"] = n
        self.checkpoints[iter_idx]["n_visible"] = int(visible.sum())

        if n == 0:
            return

        # Workload distribution
        vis_tiles = tiles_np[visible] if visible.any() else np.array([0])
        self.checkpoints[iter_idx]["tiles_mean"] = float(vis_tiles.mean())
        self.checkpoints[iter_idx]["tiles_median"] = float(np.median(vis_tiles))
        self.checkpoints[iter_idx]["tiles_max"] = float(vis_tiles.max())
        self.checkpoints[iter_idx]["tiles_p99"] = float(np.percentile(vis_tiles, 99))
        self.checkpoints[iter_idx]["tiles_p95"] = float(np.percentile(vis_tiles, 95))

        # Heavy tail: top 1% / 5% / 10% of visible Gaussians by tile count
        sorted_tiles = np.sort(vis_tiles)[::-1]
        total_tiles = sorted_tiles.sum()
        if total_tiles > 0:
            for k_pct in [1, 5, 10]:
                k = max(1, int(len(sorted_tiles) * k_pct / 100))
                mass = sorted_tiles[:k].sum() / total_tiles
                self.checkpoints[iter_idx][f"tiles_top{k_pct}_mass"] = float(mass)

        # Scale norm distribution
        self.checkpoints[iter_idx]["scale_norm_mean"] = float(scale_norm.mean())
        self.checkpoints[iter_idx]["scale_norm_median"] = float(np.median(scale_norm))

    def record_cohort_trace(self, iter_idx: int, camera_id: int,
                           tiles_per_gauss: torch.Tensor, radii: torch.Tensor,
                           ids: torch.Tensor):
        """Record per-iteration trace for cohort IDs (for lag-1 persistence)."""
        if self.cohort_ids is None:
            return

        # Find cohort IDs in current tensor
        id_to_idx = {}
        ids_cpu = ids.cpu().numpy()
        for i, gid in enumerate(ids_cpu):
            id_to_idx[int(gid)] = i

        cohort_ids_cpu = self.cohort_ids.cpu().numpy()
        trace = {}
        for cid in cohort_ids_cpu:
            if int(cid) in id_to_idx:
                idx = id_to_idx[int(cid)]
                trace[int(cid)] = {
                    "tiles": int(tiles_per_gauss[idx]),
                    "visible": bool(radii[idx].any() if radii.dim() > 1 else radii[idx] > 0),
                }

        self.cohort_trace[iter_idx] = {
            "camera_id": camera_id,
            "n_tracked": len(trace),
            "trace": trace,
        }

    def to_dict(self) -> dict:
        # Don't include full cohort trace (too large) — save separately
        return {
            "checkpoints": {str(k): v for k, v in self.checkpoints.items()},
            "cohort_size": self.cohort_size,
        }
