# Paper Workspace

This directory is reserved for the frozen, submission-facing research record.
Engineering reports under `reports/` remain useful provenance, but they are
not a substitute for a concise manuscript with pre-declared claims.

## Proposed paper focus

**Working thesis:** trainable hierarchical Gaussian splatting can reduce
end-to-end 3DGS training time by skipping work on currently irrelevant
geometry while preserving full-resolution evaluation quality.

The final title and venue should be chosen only after the confirmatory matrix
in `docs/research-readiness-audit-2026-08-05.md` is complete.

## Required files before submission

- `main.tex`: manuscript source using the target venue template.
- `claims.yaml`: frozen claim-to-artifact mapping.
- `references.bib`: verified primary-source bibliography.
- `figures/`: generated, script-traceable figures only.
- `tables/`: generated tables with source JSON paths in comments.
- `artifact-evaluation.md`: clean-machine reproduction instructions.

## Freeze rule

Once confirmatory evaluation starts, changes to the method, metric code,
scene split, seed list, stopping rule, or quality gate require a new protocol
version. Exploratory runs must not silently replace confirmatory runs.
