import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scripts.assemble_higs_paper_results import assemble  # noqa: E402


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "stats").mkdir(parents=True)
    (run / "ckpts").mkdir()
    for step, psnr in ((6999, 24.0), (14999, 26.0), (29999, 27.0)):
        (run / "stats" / f"train_step{step:04d}_rank0.json").write_text(
            json.dumps({"mem": 8.5, "ellipse_time": step * 0.05, "num_GS": 120000}),
            encoding="utf-8",
        )
    # gsplat simple_trainer evaluates at 0-indexed step = i - 1 and names the
    # val file by that step; the curve still reports 1-based protocol iterations.
    for step, psnr in ((6999, 24.1), (14999, 26.1), (29999, 27.2)):
        (run / "stats" / f"val_step{step:04d}.json").write_text(
            json.dumps({"psnr": psnr, "ssim": 0.9, "lpips": 0.1, "num_GS": 130000}),
            encoding="utf-8",
        )
    blob = b"ckpt-bytes" * 1000
    (run / "ckpts" / "ckpt_29999_rank0.pt").write_bytes(blob)
    (run / "paper-run-metadata.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "method": "higs_full",
            "scene": "mipnerf360/garden",
            "seed": 1,
            "wall_time_seconds": 3000.0,
            "started_at_utc": "2026-08-07T00:00:00+00:00",
            "hardware": {"gpu_name": "NVIDIA A100-SXM4-80GB"},
            "dataset": {"inventory_sha256": "a" * 64},
            "source": {"commit": "77ab983ffe43420b2131669cb35776b883ca4c3c"},
        }),
        encoding="utf-8",
    )
    return run


JOB = {
    "job_id": "primary_full_convergence--higs_full--mipnerf360-garden--a100--s1",
    "method": "higs_full",
    "scene": "mipnerf360/garden",
    "hardware": "a100",
    "seed": 1,
    "initialization": "from_scratch_sfm",
    "iterations": 30000,
}


def test_assemble(tmp_path):
    run = _make_run(tmp_path)
    result = assemble(run, JOB, gpu_index=3, energy_joules=1_000_000.0)
    assert result["status"] == "complete"
    assert result["job_id"] == JOB["job_id"]
    assert result["quality"]["psnr_db"] == 27.2
    assert result["performance"]["wall_time_seconds"] == 3000.0
    assert len(result["quality_curve"]) == 3
    assert [p["iteration"] for p in result["quality_curve"]] == [7000, 15000, 30000]
    walls = [p["wall_time_seconds"] for p in result["quality_curve"]]
    assert walls == sorted(walls) and len(set(walls)) == 3
    assert result["resources"]["peak_gpu_memory_mib"] == 8.5 * 1024
    assert result["resources"]["energy_joules"] == 1_000_000.0
    assert result["resources"]["final_gaussian_count"] == 130000
    assert len(result["artifact"]["sha256"]) == 64
    assert result["provenance"]["clean_process"] is True
    assert result["performance"]["time_to_quality_seconds"] <= 3000.0


def test_scheduler_plan_selects_executable_a100_gsplat_jobs(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_higs_paper_a100_matrix import _plan_jobs

    protocol_path = Path(__file__).resolve().parents[1] / "benchmark" / "higs-paper-protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    jobs = _plan_jobs(protocol, {"gsplat", "higs_full", "higs_proposed"})
    assert len(jobs) == 144
    by_method = {}
    for job in jobs:
        by_method.setdefault(job["method"], 0)
        by_method[job["method"]] += 1
        assert job["hardware"] == "a100"
        assert job["executable"] is True
        assert job["seed"] in (0, 1, 2)
    assert by_method == {"gsplat": 48, "higs_full": 48, "higs_proposed": 48}
