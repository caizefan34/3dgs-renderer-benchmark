# Leaderboard Pipeline

Generate all public tables and charts from strict Matrix v2 result records:

```text
benchmark report --output-dir docs/leaderboard
```

The output directory contains JSON, CSV, Markdown, and separate FPS-vs-PSNR and
FPS-vs-LPIPS SVG charts for measured, reproduced, and paper-reported evidence.
Tier A is preferred when available, but tiers never share a ranking table.
Overall rows require complete suite coverage in one immutable cohort.

Publication artifacts:

- [`leaderboard/ranking.md`](leaderboard/ranking.md)
- [`leaderboard/ranking.json`](leaderboard/ranking.json)
- [`leaderboard/ranking.csv`](leaderboard/ranking.csv)
- [`comparison-analysis.md`](comparison-analysis.md)

The legacy `src/scripts/generate_leaderboard.py` reads pre-Matrix artifacts only
and is not a publication path.
