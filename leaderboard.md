# Leaderboard Pipeline

Generate artifacts from benchmark JSON:

```text
python src/scripts/generate_leaderboard.py \
  --inputs data/results/rtx5070_laptop_2026-07-13.json data/results/rtx5070_train_reference_summary_2026-07-14.json \
  --output-dir results/leaderboard
```

Outputs:

- `leaderboard.json`
- `leaderboard.md`
- `leaderboard.html`

Committed GitHub Pages artifacts live in `docs/leaderboard`. Local or CI
outputs should use `results/leaderboard` unless intentionally updating Pages.

Leaderboards:

- Speed: sorted by FPS.
- Quality: sorted by PSNR, SSIM, and LPIPS among GT-scored non-synthetic rows.
- Memory: sorted by peak VRAM.
- Pareto: non-dominated rows that have compatible speed and quality metrics.

Synthetic stress rows can appear in speed and memory tables, but not in
GT-quality or quality-preserving Pareto claims.

Schema validation:

```text
python src/scripts/validate_artifacts.py --schema schemas/leaderboard.schema.json --json docs/leaderboard/leaderboard.json
```

Publication plots require matplotlib:

```text
python -m pip install -r requirements-visualization.txt
python src/scripts/generate_plots.py --inputs data/examples/evaluation_records.json --output-dir results/plots
```
