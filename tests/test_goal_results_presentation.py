from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_and_pages_surface_all_three_research_tracks():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")

    assert "## Research tracks" in readme
    assert "caizefan34.github.io/3dgs-renderer-benchmark" in readme
    assert "Three research tracks" in page
    assert "Differentiable HiGS" in page
    assert "Reproducible 3DGS survey" in page
    assert "Lossless and near-lossless storage" in page
    assert "Full convergence blocked" in page
