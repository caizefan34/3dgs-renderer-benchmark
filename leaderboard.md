# Leaderboard Pipeline

Generate all public tables and charts from strict result records:

```text
benchmark report --output-dir docs/leaderboard
```

Outputs are JSON, CSV, Markdown, and separate FPS–PSNR/FPS–LPIPS SVG charts for Measured, Reproduced, and Paper Reported evidence.
Tier A is preferred when available, but tier tables remain separate.
Overall rows require complete suite coverage.

The legacy `src/scripts/generate_leaderboard.py` remains only for reading pre-Matrix artifacts during migration.
It is not the publication path.
