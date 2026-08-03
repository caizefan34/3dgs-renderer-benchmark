"""CPU-safe tests for the error-guided tile-sampling helpers.

These test the pure-Python estimators in
``benchmark/run_higs_train_benchmark.py`` (``_tile_mean_errors``,
``_error_guided_mask``, ``_importance_l1_loss``); the CUDA isect-filtering
integration is covered by ``test_higs_frozen.py``.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

_BENCH = Path(__file__).resolve().parents[1] / "benchmark" / "run_higs_train_benchmark.py"
_spec = importlib.util.spec_from_file_location("run_higs_train_benchmark", _BENCH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_error_guided_mask = _mod._error_guided_mask
_importance_l1_loss = _mod._importance_l1_loss
_tile_mean_errors = _mod._tile_mean_errors


class TestTileMeanErrors:
    def test_exact_border_tiles(self):
        C, H, W = 2, 40, 56  # not multiples of 16 -> partial border tiles
        ts = 16
        frame = torch.zeros(1, C, H, W, 3)
        ref = torch.ones(C, H, W, 3) * 0.25
        err = _tile_mean_errors(frame, ref, ts)
        assert err.shape == (C, math.ceil(H / ts), math.ceil(W / ts))
        # interior and border tiles are all exactly 0.25 (border counted by
        # real pixels only, zero padding adds nothing to the sums)
        assert torch.allclose(err, torch.full_like(err, 0.25))


class TestImportanceL1Unbiased:
    def test_estimator_unbiased_across_draws(self):
        torch.manual_seed(7)
        C, H, W = 2, 64, 96
        ts = 16
        ref = torch.rand(C, H, W, 3)
        frame = torch.rand(1, C, H, W, 3) * 0.5
        full = (frame.squeeze(0) - ref).abs().mean()
        tile_err = _tile_mean_errors(frame, ref, ts)
        ratio = 0.5
        ests = []
        for seed in range(40):
            torch.manual_seed(seed)
            mask, weights = _error_guided_mask(tile_err, ratio, 1.0, torch.device("cpu"))
            loss = _importance_l1_loss(frame, ref, mask, weights, ts, W, H)
            ests.append(loss.item())
        mean_est = float(np.mean(ests))
        # 40 draws at r=0.5: the mean of many draws must be close to the full
        # mean (unbiased estimator) and much closer than a single draw.
        assert abs(mean_est - full.item()) < 0.02 * full.item() + 1e-4, (
            f"importance estimator biased: mean {mean_est:.5f} vs full {full.item():.5f}"
        )
        assert abs(mean_est - full.item()) < abs(ests[0] - full.item())

    def test_mask_fraction_within_ratio_and_weights_finite(self):
        torch.manual_seed(3)
        C, H, W = 1, 32, 64
        ts = 16
        tile_err = torch.rand(C, 2, 4)
        for ratio in (0.5, 0.25):
            mask, weights = _error_guided_mask(tile_err, ratio, 1.0, torch.device("cpu"))
            frac = float(mask.float().mean())
            # with-replacement draws: the unique-hit fraction is <= k/n = ratio
            # and > 0; the exact isect fraction is reported in benchmark metadata.
            assert 0.0 < frac <= ratio + 1e-6, f"mask frac {frac} outside (0, {ratio}]"
            assert torch.isfinite(weights).all() and (weights >= 0).all()