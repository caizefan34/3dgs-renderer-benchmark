# DSH EXECUTION — TFC-P0-MX

You are the research executor, not the scientific adjudicator. The user authorized **Stage P0 only** (read-only, no GPU). CVPR 2027 quality and novelty constraints are strict.

## Mission

Reconsider TFC (Tight / ellipse-exact tile footprint) as a potential *distinct* CVPR 2027 renderer systems paper. Test two independent barriers before further implementation:

- **N0 Novelty**: TFC's method must offer a precisely articulated, substantive mechanism not already covered by Speedy-Splat SnugBox/AccuTile (CVPR 2025), FlashGS (CVPR 2025), FastGS Compact Bounding Box, or gsplat upstream AccuTile PR #927 (Apr 2026). Exact ellipse-tile intersection and "move it before binning" alone FAIL N0.
- **E0 Economics / validity**: real *production* forward+backward/backward/iteration work vs genuine strong current baselines, not a component-level speedup or an analytically summed proxy. Preserve quality, gradient and training-population effects.

**Start from the historical negative/limiting evidence; don't start with a brand-new CUDA concept.**

## Stage 0A: source, provenance, freeze

1. Locate the actual repository and any MX/Epic05 TFC return artifacts. If unavailable, report `EVIDENCE_MISSING` and continue from accessible files without inventing results.
2. Capture repo HEAD, branch, `git status --porcelain`, submodules, file hashes for existing TFC, TFC_FC0, TFC2, PBR0, PUB0, PAPER1, and prior-art audit evidence. Do NOT change any input.
3. Maintain cohorts separately: EPIC05 A100-SXM4-80GB, MX A100-PCIE-40GB, local RTX5070, gsplat 1.5.3 vs upstream post-v1.5.3/AccuTile. Do not compare them as one benchmark.
4. Run portable read-only inventory script. Writes exclusively to separate output directory.
5. Inspect `PUBLICATION_CONVERGENCE_PACK_V2_1/PAPER1_OPEN_GAPS.json` BEFORE citing older `PAPER0` or PBR0/PUB0 scalars.

## Stage 0B: historical effect ledger

- TFC2 11 scenes / 88 camera records: pooled structural Gaussian–tile pair reduction 38.2%, frozen-state rendering numerically close but NOT pixel/gradient bit-exact under all test cases.
- 1.3564x legacy forward-rasterization-pipeline timing is NOT end-to-end renderer F+B or production training wall-clock.
- 1.1988x PBR0 and 1.0317x PUB0 are TIER-C component-derived proxies, not measured production iteration or 30K training.
- TFC_FC0 was listed PENDING/RUNNING in archived October-2 ledger. Search for actual newer primary outputs before deciding it remains pending. Do not assume completion just because a directory name exists.
- The TFC A-vs-B gradient difference in the hardened evidence exceeds near-deterministic A-vs-A self-noise in at least one test (e.g., cam151 0.99939 vs 0.99995), so an unconditional noise-floor explanation is disallowed.
- Separate N0 novelty from E0 runtime/quality. Even a fast prior-art mechanism is not an original paper contribution.

Produce `EVIDENCE_TIER_RECONCILIATION.md` with SOURCE / COMMIT / DATA COHORT / MEASUREMENT SCOPE / METRIC / STATUS / BIAS OR MISSING, not a merged speedup headline.

## Stage 0C: code-level prior art and original-mechanism delta

Read authoritative primary documents:
- Speedy-Splat CVPR 2025 https://openaccess.thecvf.com/content/CVPR2025/html/Hanson_Speedy-Splat_Fast_3D_Gaussian_Splatting_with_Sparse_Pixels_and_Sparse_CVPR_2025_paper.html
- Speedy-Splat project https://speedysplat.github.io/
- FlashGS CVPR 2025 https://openaccess.thecvf.com/content/CVPR2025/papers/Feng_FlashGS_Efficient_3D_Gaussian_Splatting_for_Large-scale_and_High-resolution_Rendering_CVPR_2025_paper.pdf
- gsplat main README and PR #927 https://github.com/nerfstudio-project/gsplat/pull/927
- Inspect FastGS compact bounding box original code/paper, not third-party summaries alone.

For EACH actual TFC operation, write direct code function/line, novelty claim, nearest prior-art function/equation, exact difference, computation eliminated, metadata needed, asymptotic / concrete savings, existing overlap.

**Automatic fail** for any proposal that only tightens bounding boxes, computes exact ellipse/tile intersection, reorders this operation prior to binning, or relies on existing AccuTile as if it were original.

Only keep a distinct method if source/equations establish a previously missing **execution-level mechanism** (e.g. a *new* backward- and sorting-aware interface with independently measured new work removed beyond upstream AccuTile) AND a concrete falsifier and expected expensive-work fraction. The example in parentheses is a RESEARCH QUESTION, not an authorized idea or prior-art novelty assertion.

Produce `NOVELTY_DELTA_MATRIX.md` and `CODE_LEVEL_OVERLAP.md`. If no delta, do not invent one and output `TFC_FOOTPRINT_ONLY_NO_NOVELTY`.

## Stage 0D: no-GPU production cost ceiling

At least two baseline semantics MUST be treated distinctly:
- B1 = clean frozen gsplat v1.5.3 / older round-box variant, useful historical but WEAK for 2026 novelty.
- B2 = strong upstream AccuTile-enabled gsplat and/or Speedy-Splat AccuTile implementation (version and switches pinned). Also consider Faster-GS/FlashGS separately with protocol-matched actual runs only.

Without real current-cohort measurements, record B2 comparison as `NOT_MEASURED` and do NOT compute unsupported speedup estimates. The PBR0/PUB0 proxy ceilings can motivate a cost target but are not current production measurements.

Write an Amdahl worksheet separating projection/bin/sort, raster F, raster B, loss, optimizer, and topology. If measured fractions from a single relevant production iteration are absent, leave unknown. Mark renderer-only estimates as TIER-C.

Inspect `CLOSED_DIRECTIONS_20260930.md`, `PAPER0_HARDENED`, and `PUBLICATION_CONVERGENCE_PACK_V2_1` before any new hypothesis to avoid reviving already-closed mechanisms.

## Stage 0E: predeclared verdicts

Use `DECISION_PROTOCOL.md`. Default for footprint-only = `TFC_FOOTPRINT_ONLY_NO_NOVELTY`; if source evidence inaccessible = `BLOCKED_SOURCE_EVIDENCE`; if a distinct candidate exists but economics cannot yet be established = `CONDITIONAL_RESEARCH`; ONLY if novelty delta and sufficient work fraction are properly sourced can you suggest separately authorized tiny Stage-1 tests. **Do not label P0 an implementation GO.**

## Deliverables (in a separate new directory; no source modifications)

`P0_REPORT.md`, `P0_VERDICT.json`, `EVIDENCE_TIER_RECONCILIATION.md`, `NOVELTY_DELTA_MATRIX.md`, `CODE_LEVEL_OVERLAP.md`, `STAGE1_OPTIONAL_PREREG.md` (only if non-overlap credible), `SOURCE_IDENTITY.json`, `RUN_STATUS.json`, `SHA256_MANIFEST.json`, commands & provenance.

All non-observed predictions tagged `HYPOTHESIS` / `DERIVED`, every result marked with its actual provenance; fail closed when evidence missing. No GUI code changes, no Git commit, no GPU run, no compilation, no 3K/10K/30K. Do not preempt TEMPORAL_COVERAGE-R1, WV-HIER or alter C51 archived results. Do not delete user temp files or old experiments.

## Final answer to research lead

1. What exactly is novel *beyond* Speedy-Splat, FlashGS and upstream gsplat AccuTile? Quote functions/equations.
2. Which expensive production work does the *novel* delta eliminate beyond strong baseline, with evidence?
3. Does the current TFC gradient difference require new correction/further research?
4. Is TFC_FC0 actually complete, and what exact Tier-A training metric supports it, if any?
5. What is the most economical next falsification experiment, if one exists? If no delta, STOP.
