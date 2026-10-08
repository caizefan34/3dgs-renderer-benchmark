# DetGS0 Prior-Art Sweep Log
Date: 2026-10-01
Method: arXiv Atom API (export.arxiv.org), all: fields, 3s spacing, direct HTTP.
Network notes: DuckDuckGo/Google/Bing/Semantic-Scholar unreachable from this network
(Same as VOS_PRIOR_ART.md environment note). arXiv API works.

## Queries and totalResults

| Query | totalResults | Relevant hits |
|---|---:|---|
| all:"sparse adam" AND all:"gaussian" | 1 | "A Step to Decouple Optimization in 3DGS" |
| all:"FastGS" | 5 | FastGS: Training 3DGS in 100 Seconds; +4 tangential |
| all:"GS-Scale" OR all:"GS Scale" | 4 | GS-Scale: Unlocking Large-Scale 3DGS via Host Offloading |
| all:"deterministic" AND all:"Gaussian splatting" | 39 | none on optimizer-state determinism |
| all:"exact" AND all:"optimizer" AND all:"Gaussian splatting" | 5 | none on exact optimizer state |
| all:"topology" AND all:"Gaussian splatting" AND all:"densification" | 5 | TagSplat (topology-aware mesh), none on optimizer-state exactness |
| all:"reproducibility" AND all:"Gaussian splatting" | 45 | none on optimizer-state exactness |
| all:"checkpoint" AND all:"Gaussian splatting" AND all:"optimizer state" | 0 | — |
| all:"deferred" AND all:"optimizer" AND all:"Gaussian" | 18 | GS-Scale host offloading; none on deferred optimizer state |
| all:"lazy" AND all:"optimizer" AND all:"Gaussian splatting" | 0 | — |
| all:"virtual" AND all:"optimizer state" | 4 | none relevant |
| all:"per-Gaussian" AND all:"step" AND all:"Adam" | 0 | — |
| all:"bitwise" OR all:"bit-exact" AND all:"Gaussian splatting" | 424 | mostly CS/bitwise-ops papers; none on 3DGS optimizer-state bit-exactness |
| all:"state consistency" AND all:"neural" | 41 | none relevant |
| all:"memory efficient" AND all:"Adam" AND all:"sparse" AND all:"3D" | 0 | — |

## Key findings
- ZERO public hits for "deferred/lazy/virtual per-Gaussian optimizer state with exact
  replay" in 3DGS. The closest public art is Sparse Adam (sparse step) but it is a
  MODIFIED optimizer with no dense-equivalence claim.
- ZERO public hits for "bit-exact optimizer-state preservation across densification".
- FastGS / GS-Scale are speed/scale systems; neither claims exact optimizer-state or
  topology-state consistency.
- "deterministic Gaussian splatting" hits are about rendering/SLAM, not optimizer state.
