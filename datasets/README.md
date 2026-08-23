# Local Dataset Cache

`benchmark prepare <dataset> --scene <scene>` stores verified per-scene downloads and raw files here. Large downloads, candidates, and processed assets are ignored by Git.

`benchmark prepare-case <case-id>` builds the fixed official checkpoint, 100-camera trajectory, and matching GT set. It writes `datasets/processed/<dataset>/<scene>/` only when every derived hash matches suite v3.1.0; otherwise it emits a non-ranking candidate and fails closed for Tier A.

See `docs/datasets.md` for source identities, conversion rules, and the maintainer-only audit path.
