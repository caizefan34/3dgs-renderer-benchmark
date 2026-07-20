# Command Layout

The user-facing command is `benchmark` (or `python benchmark.py` from a checkout).
Implementation scripts remain under `src/scripts/` so one CLI owns dataset preparation, execution, result collection, and report generation.

Native Ubuntu setup and the strict five-renderer Tier A runner are documented in
[`linux/README.md`](linux/README.md).
