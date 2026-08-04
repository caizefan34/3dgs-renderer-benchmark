"""CPU-safe tests for the progressive-resolution (Turbo-GS-style) helpers.

These test the pure-Python schedule/geometry helpers in
``benchmark/run_higs_train_benchmark.py`` (``_parse_res_schedule``,
``_res_stage``, ``_stage_ks``, ``_stage_refs``); the CUDA training-loop
integration is covered by the EPIC-05 runs and the existing CUDA-gated
training tests.
"""

import importlib.util
from pathlib import Path

import torch

_BENCH = Path(__file__).resolve().parents[1] / "benchmark" / "run_higs_train_benchmark.py"
_spec = importlib.util.spec_from_file_location("run_higs_train_benchmark", _BENCH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_parse_res_schedule = _mod._parse_res_schedule
_res_stage = _mod._res_stage
_stage_ks = _mod._stage_ks
_stage_refs = _mod._stage_refs


class TestParseResSchedule:
    def test_parses_pairs_in_order(self):
        assert _parse_res_schedule("0.5:0,1.0:1500") == [(0.5, 0), (1.0, 1500)]

    def test_sorts_by_start_step(self):
        assert _parse_res_schedule("1.0:1500,0.5:0") == [(0.5, 0), (1.0, 1500)]

    def test_empty_spec(self):
        assert _parse_res_schedule(None) == []
        assert _parse_res_schedule("") == []


class TestResStage:
    def test_stage_boundaries(self):
        sched = [(0.5, 0), (1.0, 1500)]
        assert _res_stage(0, sched) == 0.5
        assert _res_stage(1499, sched) == 0.5
        assert _res_stage(1500, sched) == 1.0
        assert _res_stage(2999, sched) == 1.0

    def test_fallback_without_schedule(self):
        assert _res_stage(0, [], fallback=1.0) == 1.0


class TestStageKs:
    def test_scales_focal_and_principal_point(self):
        Ks = torch.zeros(1, 2, 3, 3)
        Ks[..., 0, 0] = 1000.0
        Ks[..., 1, 1] = 1000.0
        Ks[..., 0, 2] = 959.5
        Ks[..., 1, 2] = 539.5
        K = _stage_ks(Ks, 0.5, 960, 540)
        # helper scales every camera in the [1, C, 3, 3] batch
        assert torch.allclose(K[..., 0, 0], torch.full_like(K[..., 0, 0], 500.0))
        assert torch.allclose(K[..., 1, 1], torch.full_like(K[..., 1, 1], 500.0))
        assert torch.allclose(K[..., 0, 2], torch.full_like(K[..., 0, 2], (960 - 1) / 2.0))
        assert torch.allclose(K[..., 1, 2], torch.full_like(K[..., 1, 2], (540 - 1) / 2.0))
        # original untouched (clone semantics)
        assert torch.allclose(Ks[..., 0, 0], torch.full_like(Ks[..., 0, 0], 1000.0))


class TestStageRefs:
    def test_downsample_shape(self):
        refs = torch.rand(4, 1080, 1920, 3)
        out = _stage_refs(refs, 960, 540)
        assert out.shape == (4, 540, 960, 3)
        assert out.dtype == refs.dtype

    def test_noop_when_same_shape(self):
        refs = torch.rand(2, 540, 960, 3)
        out = _stage_refs(refs, 960, 540)
        assert out is refs

class TestResScheduleFullLpipsFlag:
    """CPU-safe CLI wiring tests for --res-schedule-full-lpips.

    The flag isolates the full-res perceptual (LPIPS) signal during
    progressive-resolution coarse stages without forcing anchor-densify
    steps to full resolution (the round-52b full-signal arm, whose
    full/stage-res densify alternation destabilized high-N scenes).
    """

    def test_parser_defaults_false(self):
        parser = _mod.build_arg_parser()
        args = parser.parse_args([
            "--base-dir", "datasets/processed",
            "--scene", "mipnerf360/garden",
        ])
        assert args.res_schedule_full_lpips is False
        assert args.res_schedule_full_signal is False

    def test_parser_accepts_flag(self):
        parser = _mod.build_arg_parser()
        args = parser.parse_args([
            "--base-dir", "datasets/processed",
            "--scene", "mipnerf360/garden",
            "--res-schedule-full-lpips",
        ])
        assert args.res_schedule_full_lpips is True
        # orthogonal to the full-signal arm
        assert args.res_schedule_full_signal is False

    def test_run_backend_accepts_kwarg(self):
        import inspect
        sig = inspect.signature(_mod.run_backend)
        assert "res_schedule_full_lpips" in sig.parameters
        assert sig.parameters["res_schedule_full_lpips"].default is False


class TestMaskedPixelL1:
    """CPU-safe tests for the Speedy-Splat-style sparse-pixel loss."""

    def test_equals_full_l1_at_ratio_1(self):
        torch.manual_seed(0)
        frame = torch.rand(1, 8, 8, 3)
        ref = torch.rand(8, 8, 3)
        est = _mod._masked_pixel_l1_loss(frame, ref, 1.0, "cpu")
        full = _mod._l1_loss(frame, ref)
        assert torch.allclose(est, full, atol=1e-6)

    def test_unbiased_estimate_at_ratio_0_35(self):
        torch.manual_seed(1)
        frame = torch.rand(1, 64, 64, 3)
        ref = torch.rand(64, 64, 3)
        full = _mod._l1_loss(frame, ref).item()
        draws = [
            _mod._masked_pixel_l1_loss(frame, ref, 0.35, "cpu").item()
            for _ in range(50)
        ]
        assert abs(sum(draws) / len(draws) - full) < 0.01

    def test_empty_draw_falls_back_to_full_mean(self):
        torch.manual_seed(2)
        frame = torch.rand(1, 1, 1, 3)
        ref = torch.rand(1, 1, 1, 3)
        est = _mod._masked_pixel_l1_loss(frame, ref, 0.0, "cpu")
        assert torch.allclose(est, _mod._l1_loss(frame, ref), atol=1e-6)
