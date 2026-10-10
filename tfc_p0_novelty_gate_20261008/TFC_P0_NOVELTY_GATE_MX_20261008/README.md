# TFC-P0-MX — Exact Footprint Novelty & Production Evidence Gate

**Created**: 2026-10-08. **Execution**: NOT_EXECUTED_ON_MX. **Permission scope**: read-only evidence and source audit. No GPU, no compilation, no training, no existing file edits.

## Important finding before execution

Footprint tightening alone is **not a defensible novel CVPR contribution**:
- Speedy-Splat (CVPR 2025) proposed SnugBox + AccuTile for precise Gaussian/ellipse-to-tile assignment.
- FlashGS (CVPR 2025) covered ellipse-vs-tile tests.
- gsplat upstream lists AccuTile on the 3DGS path since Apr 2026 (PR #927; main > v1.5.3).

P0 must start with the prior-art overlap test, then inspect whether any *additional* mechanism is actually in the TFC code. Do not automatically declare all potential future work impossible; declare **TFC_FOOTPRINT_ONLY_NO_NOVELTY** unless a specifically different, evidenced method survives.

## Archive order

1. `DSH_EXECUTION_PROMPT.md`: ready-to-forward instruction for DSH.
2. `PRIOR_ART_AND_DELTA.md`: source-level novelty comparison requirements.
3. `LEGACY_EVIDENCE_LEDGER.json`: performance claim tiers and supersessions.
4. `DECISION_PROTOCOL.md`: fail-closed gates; what may follow P0.
5. `scripts/stage0_readonly_inventory.py`: portable source/status/hash inventory. Writes only to an external output directory.
6. `scripts/verify_manifest.py`: verify the package manifest.
7. `evidence/REPO_SOURCE_URLS.txt`: precise existing repository files, not local MX paths.

## Launch examples

Linux/MX (only when repository and evidence are locally available):

```bash
python3 scripts/verify_manifest.py
python3 scripts/stage0_readonly_inventory.py --repo /path/to/3dgs-renderer-benchmark --out /tmp/tfc_p0_inventory_20261008
```

Windows:

```powershell
python scripts/verify_manifest.py
python scripts/stage0_readonly_inventory.py --repo 'D:\path\to\3dgs-renderer-benchmark' --out "$env:TEMP\tfc_p0_inventory_20261008"
```

The inventory is an **audit**; it does not benchmark or certify the design. If source files are not present, report `EVIDENCE_MISSING`, not PASS.
