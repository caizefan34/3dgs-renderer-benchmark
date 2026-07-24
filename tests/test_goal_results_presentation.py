from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_and_pages_surface_all_three_optimization_goals():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")

    assert "Optimization goals and delivery status" in readme
    assert "caizefan34.github.io/3dgs-renderer-benchmark" in readme
    assert "Three optimization goals: delivered evidence" in page
    assert "Why HiGS is inference-only" in page
    assert "Fuse renderer acceleration ideas into HiGS" in page
    assert "Maximum near-lossless checkpoint compression" in page
