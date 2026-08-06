# 3DGS survey protocol

## Review question

Which publicly described 3DGS methods materially change rendering throughput,
training cost, or model storage, and which of them can be reproduced under a
common artifact protocol?

The current registry is a reproducibility map, not yet a systematic review.
The phrase "latest survey" may be used only with a visible search date and the
completed process below.

## Scope and taxonomy

Include work whose primary contribution affects at least one of:

- renderer algorithms, ordering, culling, precision, or hardware mapping;
- differentiable rasterization, backward computation, or training schedules;
- checkpoint representation, quantization, entropy coding, or learned storage.

Tag each work by task, approximation class, retraining requirement, temporal
state, supported hardware, source availability, license, and evidence tier.
Do not compare paper-reported speedups numerically across different cohorts.

## Search protocol required for a systematic claim

Freeze the following before manuscript submission:

1. Search date and databases: arXiv, Google Scholar, Semantic Scholar, IEEE
   Xplore, ACM Digital Library, CVF Open Access, and citation graphs of included
   papers.
2. Exact queries, including combinations of `3D Gaussian Splatting` with
   `rendering`, `rasterization`, `training`, `backward`, `compression`,
   `quantization`, `streaming`, and `storage`.
3. Deduplicated candidate list with title, authors, venue, year, URL, and source
   repository.
4. Two-stage screening: title/abstract, then full text and artifact.
5. Inclusion and exclusion reason for every full-text candidate.
6. A second-reviewer audit of all exclusions and a sample of inclusions.

## Evidence tiers

| Tier | Meaning | Permitted use |
| --- | --- | --- |
| Reported | Number copied from a paper or upstream repository | Context only; retain the original cohort |
| Reproduced | Official code executed with pinned inputs and environment | Reproduction table, separate from the common benchmark |
| Comparable | Same protocol, assets, hardware cohort, and quality gate | Within-cohort ranking and statistical analysis |
| Blocked | Source, build, dataset, or protocol requirement missing | Coverage table with a dated reason |

## Update and authority policy

Publish the search date, registry version, and changelog. Invite method authors
to verify metadata and adapters through reviewable issues or pull requests.
Never silently replace an upstream result or remove a negative reproduction.
Authority comes from traceable decisions and reproducible evidence, not from a
claim of exhaustive coverage.
