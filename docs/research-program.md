# Research program

This repository studies 3D Gaussian Splatting through three related but
independently publishable tracks. The shared infrastructure provides pinned
sources, fixed scenes, provenance, quality metrics, and evidence validation.
It does not make the three tracks one scientific contribution.

| Track | Scientific question | Current evidence | Paper boundary |
| --- | --- | --- | --- |
| Survey | Which open 3DGS methods are reproducible, comparable, and relevant to rendering, training, or storage? | Ten-family source-pinned renderer registry and a five-renderer A100 matrix | Systematic-review claims remain blocked until the search and screening log is frozen |
| Differentiable HiGS | Can hierarchical rendering support correct gradients and reduce end-to-end training cost without sacrificing converged quality? | Native CUDA backward, topology lifecycle, tracked multi-seed short-horizon studies, and extensive ablations | Full-convergence and official-baseline claims remain blocked |
| Storage compression | Which same-checkpoint format minimizes storage subject to a declared quality gate? | Five-scene round-trip comparison of bit-exact and near-lossless formats | Learned codecs that require retraining form a separate cohort |

## Recommended publication strategy

The HiGS method is the strongest candidate for a primary method paper once its
full-training gates are complete. The survey can become an artifact or survey
paper after a systematic search audit. The storage work can become an empirical
study after adding decode and deployment costs plus current learned codecs.

Do not combine all three as co-equal contributions in one full paper. A shared
repository and artifact is useful; a manuscript still needs one central causal
question, no more than three contributions, and one claims manifest.

## Evidence contract

Each track has a machine-readable manifest in `paper/`:

- `survey-claims.json`
- `higs-claims.json`
- `compression-claims.json`

Run all three gates with:

```bash
python src/scripts/validate_research_program.py
```

`supported` claims are bound to Git-tracked JSON by SHA-256 and executable
assertions. `blocked` claims name the experiment or review step still required.
`out_of_scope` prevents an attractive but invalid universal claim from entering
the abstract or conclusion.

## Shared academic rules

1. Keep literature-reported, locally reproduced, and newly proposed results in
   different evidence tiers.
2. Never mix hardware, checkpoints, cameras, resolutions, timing boundaries, or
   retraining requirements in one ranking.
3. Report raw repeats and uncertainty. Frames are workload samples, not
   independent experimental replicates.
4. Preserve negative results and exclusion reasons, but move chronological logs
   out of the main paper narrative.
5. Use stars, citations, and downloads only as adoption indicators, never as
   evidence of scientific correctness.
