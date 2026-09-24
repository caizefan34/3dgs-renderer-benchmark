#!/usr/bin/env python3
"""Phase C53-Validation2 — Ranking Comparison (M1/M2/M3 Recall@K). See analyze_all.py for full analysis."""
import subprocess, sys
subprocess.run([sys.executable, str(__import__('pathlib').Path(__file__).parent / "analyze_all.py")])
