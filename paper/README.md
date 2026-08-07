# Paper evidence and submission workspace

This directory is the submission-facing research record. Engineering reports under `reports/` remain useful provenance, but they are not a substitute for a concise manuscript with pre-declared claims.

## Manuscript scopes and frozen manifests

Four manuscript scopes are kept separate. [`claims.json`](claims.json) is the frozen renderer-benchmark artifact paper. The repository research program has three additional, independently publishable tracks:

| Track | Manifest | Current role |
| --- | --- | --- |
| Reproducible survey | [`survey-claims.json`](survey-claims.json) | Registry supported; systematic-search claims blocked |
| Differentiable HiGS | [`higs-claims.json`](higs-claims.json) | Native backward + trainability + memory reduction supported; quality-preserving speedup blocked |
| Storage compression | [`compression-claims.json`](compression-claims.json) | Five-scene SPZ qualification supported; deployment costs blocked |

Every manifest contains at most three contributions and maps supported statements to Git-tracked, SHA-256-pinned evidence plus machine-checkable assertions. The tracks share an artifact but should not be merged into one paper as co-equal contributions.

## HiGS working thesis

**Working thesis:** trainable hierarchical Gaussian splatting can reduce end-to-end 3DGS training cost (memory and wall time) while preserving full-resolution evaluation quality. The frozen 177-job from-scratch A100 matrix now supports the trainability and memory-reduction components; the full-convergence quality-preserving speedup component remains a blocked claim whose named gates are listed in [`higs/README.md`](higs/README.md).

## Required files before submission

- `main.tex`: manuscript source using the target venue template.
- `claims.yaml` / `claims.json`: frozen claim-to-artifact mapping.
- `references.bib`: verified primary-source bibliography.
- `figures/`: generated, script-traceable figures only.
- `tables/`: generated tables with source JSON paths in comments.
- `artifact-evaluation.md`: clean-machine reproduction instructions.

## Validation and release

```bash
python src/scripts/validate_paper_evidence.py          # frozen benchmark-paper manifest
python src/scripts/validate_research_program.py       # survey + HiGS + compression tracks
python src/scripts/build_release_bundle.py --output 3dgs-renderer-benchmark-<version>.zip
```

On any `v*` tag, the `release-evidence-bundle` workflow revalidates the claims, builds the bundle, and creates a GitHub Release with it. Archive that zip to Zenodo and then add the issued DOI to `CITATION.cff`.

A `supported` claim may enter the abstract, tables, or conclusions. A `blocked` claim must not be written as a result until its named gate is completed. `out_of_scope` keeps attractive but distracting claims out of this paper.

## Freeze rule

Once confirmatory evaluation starts, changes to the method, metric code, scene split, seed list, stopping rule, or quality gate require a new protocol version. Exploratory runs must not silently replace confirmatory runs.

