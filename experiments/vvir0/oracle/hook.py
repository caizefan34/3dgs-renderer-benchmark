"""
VRIR0 oracle hook — captures fresh raster structure and evaluates certificates.

The baseline still executes the normal fresh intersection/sort/offset path every
iteration (this module never skips it). On each tracked-camera visit we:
  1. capture current fresh meta (means2d/radii/depths/flatten_ids/isect_offsets);
  2. compute membership descriptor from verified semantics;
  3. find previous cache entry for same camera;
  4. evaluate CONFIG/TOPOLOGY/NUMERIC/MEMBERSHIP/ORDER certificates;
  5. independently compare cached structure with current fresh structure;
  6. append one JSONL record;
  7. replace cached entry with current fresh structure.

Large structure tensors are stored on CPU. A byte cap is enforced; exceeding it
records CACHE_BUDGET_REJECT instead of dropping data silently.
"""
from __future__ import annotations
import json, os, time
from collections import defaultdict
from typing import Dict, Optional, List
import numpy as np
import torch

try:
    from .certificate import (Structure, tile_bounds, evaluate_certificate,
                              build_key_slack_budgets, validate_membership_model)
except ImportError:
    from certificate import (Structure, tile_bounds, evaluate_certificate,
                             build_key_slack_budgets, validate_membership_model)

class VRIR0Oracle:
    def __init__(self, scene: str, width: int, height: int, tile_size: int,
                 tracked_views: List[int], cache_byte_cap: int,
                 record_path: str, epoch_fn=None):
        self.scene = scene
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.tracked_views = set(int(v) for v in tracked_views)
        self.cache_byte_cap = int(cache_byte_cap)
        self.record_path = record_path
        self.epoch_fn = epoch_fn  # callables -> current topology_epoch
        self._cache: Dict[int, Structure] = {}
        self._cache_bytes = 0
        self._revisit_count = 0
        self._first_visit_count = 0
        self._cache_rejects = 0
        self._records: List[dict] = []
        os.makedirs(os.path.dirname(record_path), exist_ok=True)

    def _structure_bytes(self, s: Structure) -> int:
        return (s.flatten_ids.nbytes + s.isect_offsets.nbytes + s.means2d.nbytes
                + s.radii.nbytes + s.depths.nbytes + s.b_minus.nbytes
                + s.b_plus.nbytes + s.active.nbytes + s.depth_bits.nbytes)

    def _capture(self, meta: dict, iteration: int, camera_id: int,
                 topology_epoch: int) -> Structure:
        m2d = meta['means2d'][0].detach().to('cpu', torch.float64).numpy()
        radii = meta['radii'][0].detach().to('cpu', torch.float64).numpy()
        depths = meta['depths'][0].detach().to('cpu', torch.float64).numpy()
        b_minus, b_plus, active = tile_bounds(m2d, radii, self.width, self.height, self.tile_size)
        # gsplat leaves means2d/depths uninitialized for culled gaussians; zero inactive
        # entries so no NaN/garbage propagates into depth_bits or drift diagnostics.
        if not bool(np.all(active)):
            m2d = m2d.copy(); depths = depths.copy()
            m2d[~active] = 0.0; depths[~active] = 0.0
        # depth bits from float32 view-space z (only meaningful for active)
        dep32 = depths.astype('<f4')
        depth_bits = dep32.view('<u4').astype(np.uint32)
        flat = meta['flatten_ids'].detach().to('cpu', torch.int32).numpy().copy()
        # isect_offsets is [C, tile_height, tile_width]: per-tile start offset into flat.
        # Flatten camera-0 row-major (tile_id = ty*tw + tx, matching the sort order) into a
        # 1D exclusive prefix-sum over tiles and append n_isects as the terminal offset.
        off_raw = meta['isect_offsets'][0].detach().to('cpu', torch.int32).numpy()
        off = np.concatenate([off_raw.reshape(-1),
                              np.array([flat.shape[0]], dtype=np.int32)]).astype(np.int32)
        return Structure(
            camera_id=int(camera_id), iteration=int(iteration),
            width=self.width, height=self.height,
            n_gaussians=int(m2d.shape[0]),
            flatten_ids=flat, isect_offsets=off,
            tile_width=(self.width + self.tile_size - 1)//self.tile_size,
            tile_height=(self.height + self.tile_size - 1)//self.tile_size,
            means2d=m2d, radii=radii, depths=depths,
            b_minus=b_minus, b_plus=b_plus, active=active,
            depth_bits=depth_bits, topology_epoch=int(topology_epoch),
            n_isects=int(flat.shape[0]),
        )

    def on_visit(self, meta: dict, iteration: int, camera_id: int,
                 config_same: bool) -> Optional[dict]:
        """Called every iteration. Only acts on tracked-camera revisits."""
        if int(camera_id) not in self.tracked_views:
            return None
        epoch = self.epoch_fn() if self.epoch_fn is not None else 0
        fresh = self._capture(meta, iteration, camera_id, epoch)
        # P3 stop criterion: computed membership descriptor must match fresh structure.
        try:
            mm_ok, mm_bad = validate_membership_model(
                fresh.flatten_ids, fresh.isect_offsets, fresh.b_minus, fresh.b_plus,
                fresh.active, fresh.tile_width, fresh.n_gaussians)
        except Exception:
            mm_ok, mm_bad = False, -1
        cached = self._cache.get(int(camera_id))
        if cached is None:
            self._first_visit_count += 1
        else:
            self._revisit_count += 1
        res = evaluate_certificate(cached, fresh, config_same=config_same, epoch=epoch)
        rec = {
            'scene': self.scene,
            'iteration': int(iteration),
            'camera_id': int(camera_id),
            'cache_age_iters': (int(iteration) - int(cached.iteration)) if cached is not None else 0,
            'topology_epoch': int(epoch),
            'eligible': cached is not None,
            'certificate': {
                'config': bool(res.config),
                'topology': bool(res.topology),
                'numeric': bool(res.numeric),
                'membership': bool(res.membership),
                'order_exact_scan': bool(res.order_exact_scan),
                'order_slack_n': bool(res.order_slack_n),
                'accept': bool(res.accept),
            },
            'reject_reason': res.reject_reason,
            'truth_same': None,  # filled below (needs ground-truth compare)
            'false_accept': bool(res.false_accept),
            'false_reject': bool(res.false_reject),
            'tile_truth_same_fraction': (None if np.isnan(res.tile_truth_same_fraction) else float(res.tile_truth_same_fraction)),
            'membership_model_ok': bool(mm_ok),
            'membership_model_mismatch': int(mm_bad),
            'adjacent_order_violations': int(res.adjacent_order_violations),
            'adjacent_pairs_checked': int(res.adjacent_pairs_checked),
            'n_isects': int(fresh.n_isects),
            'cache_bytes': int(self._cache_bytes),
            'drift': self._drift(cached, fresh),
        }
        # ground-truth same flag
        if cached is not None:
            tsame, _ = _truth_same_tuple(cached, fresh)
            rec['truth_same'] = bool(tsame)
        # append JSONL
        with open(self.record_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, separators=(',', ':')) + '\n')
        self._records.append(rec)
        # replace cache with current fresh structure (enforce byte cap)
        self._update_cache(int(camera_id), fresh)
        return rec

    def _update_cache(self, camera_id: int, fresh: Structure):
        old = self._cache.get(camera_id)
        if old is not None:
            self._cache_bytes -= self._structure_bytes(old)
        nb = self._structure_bytes(fresh)
        if self._cache_bytes + nb > self.cache_byte_cap:
            self._cache_rejects += 1
            # record budget reject; do not silently drop. Keep old cache.
            rec = {'scene': self.scene, 'iteration': int(fresh.iteration),
                   'camera_id': camera_id, 'event': 'CACHE_BUDGET_REJECT',
                   'needed_bytes': int(nb), 'cache_bytes': int(self._cache_bytes),
                   'cap': int(self.cache_byte_cap)}
            with open(self.record_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, separators=(',', ':')) + '\n')
            return
        self._cache[camera_id] = fresh
        self._cache_bytes += nb

    def _drift(self, cached: Optional[Structure], fresh: Structure) -> Optional[dict]:
        if cached is None:
            return None
        assert cached.n_gaussians == fresh.n_gaussians
        d_m2d = np.abs(fresh.means2d - cached.means2d)
        d_r = np.abs(fresh.radii - cached.radii)
        act = fresh.active
        d_dep = np.abs(fresh.depths - cached.depths)
        def q(a):
            if a.size == 0: return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0}
            return {'p50': float(np.percentile(a, 50)),
                    'p95': float(np.percentile(a, 95)),
                    'p99': float(np.percentile(a, 99))}
        return {'means2d': q(d_m2d[act].reshape(-1)) if np.any(act) else q(d_m2d.reshape(-1)),
                'radii': q(d_r[act].reshape(-1)) if np.any(act) else q(d_r.reshape(-1)),
                'depth': q(d_dep[act].reshape(-1)) if np.any(act) else q(d_dep.reshape(-1))}

    def summary(self) -> dict:
        return {'revisit_count': self._revisit_count,
                'first_visit_count': self._first_visit_count,
                'cache_rejects': self._cache_rejects,
                'cache_bytes': self._cache_bytes,
                'n_records': len(self._records)}

def _truth_same_tuple(cached: Structure, fresh: Structure):
    try:
        from .certificate import truth_same
    except ImportError:
        from certificate import truth_same
    return truth_same(cached, fresh)
