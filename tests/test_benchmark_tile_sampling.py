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


class TestErrorGuidedLambdaMix:
    def test_lambda_zero_estimator_still_unbiased(self):
        torch.manual_seed(11)
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
            mask, weights = _error_guided_mask(
                tile_err, ratio, 1.0, torch.device("cpu"), lambda_mix=0.0
            )
            loss = _importance_l1_loss(frame, ref, mask, weights, ts, W, H)
            ests.append(loss.item())
        mean_est = float(np.mean(ests))
        # uniform-mix (lambda=0) still gives an unbiased estimator: the mean of
        # many draws must approach the full mean and beat a single draw.
        assert abs(mean_est - full.item()) < 0.02 * full.item() + 1e-4, (
            f"lambda=0 estimator biased: mean {mean_est:.5f} vs full {full.item():.5f}"
        )
        assert abs(mean_est - full.item()) < abs(ests[0] - full.item())

    def test_lambda_zero_mask_fraction_and_weights(self):
        torch.manual_seed(13)
        C, H, W = 1, 32, 64
        ts = 16
        tile_err = torch.rand(C, 2, 4) * 10  # strongly non-uniform
        ratio = 0.5
        mask, weights = _error_guided_mask(
            tile_err, ratio, 1.0, torch.device("cpu"), lambda_mix=0.0
        )
        frac = float(mask.float().mean())
        assert 0.0 < frac <= ratio + 1e-6, f"mask frac {frac} outside (0, {ratio}]"
        assert torch.isfinite(weights).all() and (weights >= 0).all()
        # p = uniform -> weights = m * n / k, i.e. integer counts scaled by n/k
        n = tile_err.shape[1] * tile_err.shape[2]
        k = int(round(n * ratio))
        cnt = weights * k / n
        assert torch.allclose(cnt, cnt.round(), atol=1e-5), "uniform-mix weights not m*n/k"
        assert (cnt >= 0).all() and (cnt[mask] > 0).all()

    def test_lambda_one_is_default_unchanged(self):
        torch.manual_seed(17)
        tile_err = torch.rand(1, 2, 4) * 5
        ratio = 0.5
        torch.manual_seed(17)
        m1, w1 = _error_guided_mask(tile_err, ratio, 1.0, torch.device("cpu"))
        torch.manual_seed(17)
        m2, w2 = _error_guided_mask(
            tile_err, ratio, 1.0, torch.device("cpu"), lambda_mix=1.0
        )
        assert torch.equal(m1, m2) and torch.allclose(w1, w2)


class TestLrSchedule:
    def test_endpoints_match_exponential_decay(self):
        base, decay, steps = 1.6e-4, 0.1, 3000
        lr0 = _mod._lr_at_step(base, decay, 0, steps)
        lr_last = _mod._lr_at_step(base, decay, steps - 1, steps)
        assert lr0 == pytest.approx(base * decay ** (1.0 / steps))
        assert lr_last == pytest.approx(base * decay)  # final step reaches base*decay

    def test_monotone_non_increasing(self):
        base, decay, steps = 1e-3, 0.1, 300
        vals = [_mod._lr_at_step(base, decay, it, steps) for it in range(steps)]
        assert all(a >= b for a, b in zip(vals, vals[1:]))
        assert vals[0] > vals[-1]

    def test_no_decay_is_constant(self):
        for it in (0, 7, 299):
            assert _mod._lr_at_step(5e-3, 1.0, it, 300) == 5e-3


class TestLpipsLoss:
    def test_normalize_shape_and_range(self):
        torch.manual_seed(3)
        frame = torch.rand(1, 2, 8, 8, 3)
        ref = torch.rand(2, 8, 8, 3)
        xa, ya = _mod._lpips_normalize(frame, ref)
        assert xa.shape == (2, 3, 8, 8) and ya.shape == (2, 3, 8, 8)
        assert xa.min() >= -1.0 - 1e-5 and xa.max() <= 1.0 + 1e-5
        assert ya.min() >= -1.0 - 1e-5 and ya.max() <= 1.0 + 1e-5

    def test_normalize_identity_is_zero_input(self):
        torch.manual_seed(3)
        x = torch.rand(1, 2, 8, 8, 3)
        xa, ya = _mod._lpips_normalize(x, x.squeeze(0))
        assert torch.allclose(xa, ya)

    def test_cli_args_exposed(self):
        ap = _mod.build_arg_parser()
        ns = ap.parse_args(
            ["--lpips-loss-weight", "0.1", "--lpips-loss-every", "25"]
        )
        assert ns.lpips_loss_weight == 0.1 and ns.lpips_loss_every == 25
        assert ns.lr_decay == 1.0 and ns.densify_window == 0  # defaults
