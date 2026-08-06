import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.assemble_higs_paper_results import assemble  # noqa: E402
from scripts.build_higs_paper_tables import main as build_main  # noqa: E402


def _result(run_dir: Path, seed: int, psnr: float, wall: float) -> dict:
    (run_dir / "stats").mkdir(parents=True, exist_ok=True)
    (run_dir / "ckpts").mkdir(exist_ok=True)
    for step, t in ((6999, 300.0), (14999, 900.0), (29999, wall - 50.0)):
        (run_dir / "stats" / f"train_step{step:04d}_rank0.json").write_text(
            json.dumps({"mem": 3.0, "ellipse_time": t, "num_GS": 120000}), encoding="utf-8"
        )
    for step, p in ((6999, psnr - 2.0), (14999, psnr - 1.0), (29999, psnr)):
        (run_dir / "stats" / f"val_step{step:04d}.json").write_text(
            json.dumps({"psnr": p, "ssim": 0.88, "lpips": 0.12, "num_GS": 130000}), encoding="utf-8"
        )
    (run_dir / "ckpts" / "ckpt_29999_rank0.pt").write_bytes(b"x" * 100)
    (run_dir / "paper-run-metadata.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "method": "higs_full",
            "scene": "mipnerf360/garden",
            "seed": seed,
            "wall_time_seconds": wall,
            "started_at_utc": "2026-08-07T00:00:00+00:00",
            "hardware": {"gpu_name": "NVIDIA A100-SXM4-80GB"},
            "dataset": {"inventory_sha256": "a" * 64},
            "source": {"commit": "77ab983ffe43420b2131669cb35776b883ca4c3c"},
        }),
        encoding="utf-8",
    )
    return assemble(run_dir, {
        "job_id": f"primary_full_convergence--higs_full--mipnerf360-garden--a100--s{seed}",
        "method": "higs_full",
        "scene": "mipnerf360/garden",
        "hardware": "a100",
        "seed": seed,
        "initialization": "from_scratch_sfm",
        "iterations": 30000,
    }, gpu_index=0, energy_joules=1_000_000.0)


def test_build_higs_paper_tables(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for seed, psnr, wall in ((0, 27.0, 1200.0), (1, 27.1, 1250.0), (2, 26.9, 1180.0)):
        run_dir = tmp_path / f"run{seed}"
        result = _result(run_dir, seed, psnr, wall)
        (results_dir / f"{result['job_id']}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    output = tmp_path / "tables"
    rc = build_main(["--results-dir", str(results_dir), "--output-dir", str(output)])
    assert rc == 0
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "higs_full" in summary
    assert "27.00 +/- 0.10" in summary
    conv = (output / "convergence.csv").read_text(encoding="utf-8")
    assert conv.count("\n") == 10  # header + 3 seeds x 3 curve points
    aggregate = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["scenes"] == 1
    summary_json = json.loads((output / "matrix-summary.json").read_text(encoding="utf-8"))
    assert summary_json["report"]["complete"] == 3
    assert summary_json["per_method"]["higs_full"]["jobs"] == 3
    assert summary_json["per_method"]["higs_full"]["peak_gpu_memory_mib_mean"] > 0


def test_build_rejects_invalid_result(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    bad = {"job_id": "not-in-plan", "status": "complete"}
    (results_dir / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    try:
        build_main(["--results-dir", str(results_dir), "--output-dir", str(tmp_path / "tables")])
        assert False, "expected SystemExit for invalid result"
    except SystemExit as exc:
        assert exc.code != 0

def test_build_requires_full_matrix_when_flag_set(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    run_dir = tmp_path / "run0"
    result = _result(run_dir, 0, 27.0, 1200.0)
    (results_dir / f"{result['job_id']}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    try:
        build_main(["--results-dir", str(results_dir), "--output-dir", str(tmp_path / "tables"), "--require-complete"])
        assert False, "expected SystemExit for incomplete matrix"
    except SystemExit as exc:
        assert exc.code != 0
