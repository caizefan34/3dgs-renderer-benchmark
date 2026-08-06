# Paper evidence packages

This directory separates four manuscript scopes. [`claims.json`](claims.json)
is the frozen renderer-benchmark artifact paper. The repository research
program has three additional, independently publishable tracks:

| Track | Manifest | Current role |
| --- | --- | --- |
| Reproducible survey | [`survey-claims.json`](survey-claims.json) | Registry supported; systematic-search claims blocked |
| Differentiable HiGS | [`higs-claims.json`](higs-claims.json) | Short-horizon multi-seed artifact supported; full convergence blocked |
| Storage compression | [`compression-claims.json`](compression-claims.json) | Five-scene SPZ qualification supported; deployment costs blocked |

Every manifest contains at most three contributions and maps supported
statements to Git-tracked, SHA-256-pinned evidence plus machine-checkable
assertions. The tracks share an artifact but should not be merged into one paper
as co-equal contributions.

Validate it with:

```bash
python src/scripts/validate_paper_evidence.py
```

Validate all three research tracks with:

```bash
python src/scripts/validate_research_program.py
```

The HiGS top-conference experiment contract and manuscript map are in
[`higs/README.md`](higs/README.md). Its full-training matrix is validated
separately from the short-horizon research artifacts.

Build the deterministic, self-verifying archival bundle with:

```bash
python src/scripts/build_release_bundle.py --output 3dgs-renderer-benchmark-<version>.zip
```

This bundle currently archives the frozen benchmark-paper manifest. The HiGS,
survey, and compression manuscripts require their own frozen release bundle
after their blocked submission gates are completed; do not cite the benchmark
bundle as if it archived unfinished method claims.

On any `v*` tag, the `release-evidence-bundle` workflow revalidates the claims,
builds the bundle, and creates a GitHub Release with it. Archive that zip to
Zenodo and then add the issued DOI to `CITATION.cff`.

A `supported` claim may enter the abstract, tables, or conclusions. A `blocked`
claim must not be written as a result until its named gate is completed.
`out_of_scope` keeps attractive but distracting claims out of this paper.

The benchmark scope remains deliberately narrow: selected renderers and codecs
on one five-scene NVIDIA A100 cohort. The HiGS manifest does not yet claim full
from-scratch convergence. The survey does not yet claim exhaustive coverage.
The compression manifest does not equate near-lossless with bit-exact lossless.
