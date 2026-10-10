# COMMANDS AND PROVENANCE — TFC-P0-MX

All actions this task were **read-only**. No GPU, no compilation, no 3K/10K/30K run, no source modification, no Git commit, no deletion.

## Environment
- Working dir: `C:\Users\36570\3dgs-renderer-benchmark`
- OS/shell: Windows, PowerShell (`pwsh`)
- GPU used: **none** (0 GPU-seconds)

## Read-only operations performed
1. Read protocol inputs under `tfc_p0_novelty_gate_20261008/TFC_P0_NOVELTY_GATE_MX_20261008/`:
   - `DSH_EXECUTION_PROMPT.md`, `DECISION_PROTOCOL.md`, `PRIOR_ART_AND_DELTA.md`, `README.md`, `evidence/LEGACY_EVIDENCE_LEDGER.json`, `evidence/REPO_SOURCE_URLS.txt`, `scripts/stage0_readonly_inventory.py`, `scripts/verify_manifest.py`.
2. Read prior-art audit artifacts under `_tfc_prior_art_audit/`:
   - `TFC_VS_ACCUTILE_MEMBERSHIP.json`, `TFC_CURRENT_GSPLAT_BASELINE_AUDIT.json`, `TFC_COMPLEXITY_OPERATION_AUDIT.json`, `TFC_HEAD_TO_HEAD_TIMING.json`, `TFC_PRIOR_ART_VERDICT.json`, `TFC_VS_ACCUTILE_SOURCE_DIFF.json`, `TFC_VS_ACCUTILE_MEMBERSHIP.json`.
3. Read training/gradient evidence: `_tfc2_oracle/TFC2_VERDICT.json`, `_pbr0_oracle/PBR0_H1_HARNESS_AUDIT.json`, `PBR0_VERDICT.json`, `PBR0_TIGHTENING_ORACLE.json`, `_tfc_fc0_pack/deliverables/*` (TIMING/QUALITY/VERDICT/SOURCE_LOCK/GAUSSIAN_COUNT/REPORT), `_tfc_fc0_pack/parity_packed.py`, `_tfc_fc0_pack/bwd_test.py`.
4. Read strong-baseline (AccuTile) evidence: `reports/accutile/accutile-a100-final.md`, `accutile-b1a-r6a-rescreen.md`, `accutile-a100-raw-results.json`; `reports/accutile30k/results.json`, `final-verdict.md`, `13scene-training-speed.md`, `13scene-quality.md`, `run-manifest.md`, `b1-reuse-audit.md`; `reports/final_sprint/accutile_source_audit.md`, `accutile_correctness.md`, `accutile_forensics.md`, `accutile_opportunity_gate.md`.
5. Read upstream local AccuTile source: `public_baselines/gsplat/gsplat/cuda/csrc/IntersectTile.cu` (current-main, sha `1f331b6d…`).
6. Read closed-directions pre-screen: `CLOSED_DIRECTIONS_20260930.md`; `reports/phase-c17-c2/prior_art_boundary_audit.md`.
7. Inspected publication hierarchy: `PUBLICATION_CONVERGENCE_PACK_V2_1/PAPER1_OPEN_GAPS.json`, `PAPER1_EVIDENCE_MATRIX.json`, `PAPER1_REPORT.md`, `FINAL_PROTOCOL_LOCK.md` (Stage 0A.5 requirement: inspect before citing older scalars).

## Hash computations (read-only, Get-FileHash SHA256)
```
Get-FileHash -Algorithm SHA256 <path>
```
Live results (also recorded in `SOURCE_IDENTITY.json`):
- `_tfc1_oracle/src/tfc1_rowwise.cu` = `a44e4ee63c7ba9d713a4489693d189cb3fd6e7ba95764e491027c0639e34e68e` (== freeze lock, intact)
- `_tfc2_oracle/src/tfc2_ablation.cu` = `3c9b2e0a1deda28fa1521f88b054a1b670d843b1b001a12c4b1d8b5fe4e5dd81`
- `public_baselines/gsplat/gsplat/cuda/csrc/IntersectTile.cu` = `1f331b6dc73cf16f1e496f4a525f3abec3ef1bc132b126408f04a46f2599e712` (== current-main commit 512d366)

## Output archive
- Directory: `tfc_p0_novelty_gate_20261008/P0_OUTPUT_MX_20261008/` (new, independent).
- Inputs were **not** written to; only this new directory was created.

## Provenance caveats
- No `git` command was issued; HEAD/branch/`git status --porcelain` were **not** captured live. Historical commit hashes are quoted from artifact provenance fields (`TFC_FC0_SOURCE_LOCK.json`, `SOURCE_IDENTITY.json`, `reports/final_sprint/accutile_source_audit.md`).
- The membership test in `TFC_VS_ACCUTILE_MEMBERSHIP.json` was executed on the MX cohort under GPU contention (3 procs/GPU); it is a deterministic set-equality test with **timing explicitly NOT issued** — used only for structural (not timing) claims.
