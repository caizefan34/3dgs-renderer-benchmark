# VRIR0 Prior-Art Exclusion Matrix (freeze date: 2026-10-01)

The novelty search must be refreshed immediately before any paper claim.

| Work | Setting / changing axis | Reuse object | Training? | Exact structural certificate? | VRIR0 overlap / hazard |
|---|---|---|---|---|---|
| Lumina / S² (ISCA 2025) | fixed trained scene, adjacent moving camera poses | previous sorting result across frames; radiance cache | no | no; tolerates rare local order changes | Direct hazard to any claim of “first sorting reuse” or “temporal sorting sharing” |
| Neo (ASPLOS 2026) | fixed trained scene, consecutive inference frames | reuse-and-update Gaussian depth sorting; handles reorder/insert/delete | no | update logic, not training-state fail-closed certification | Direct hazard to incremental sorting reuse novelty |
| CaT-GS (CVPR 2026) | fixed scene, nearby frames in a speculative batch | visible-set/frustum cache and cached sorted/list-generation output | no | no exact current-state proof; accepts controlled redundancy | Direct hazard to cached tile/list and skip-sort novelty |
| GSReuse (TVCG/VR 2026) | fixed scene, consecutive frames | screen-space warped reusable content + tile filtering | no | approximate/quality-preserving | Hazard to broad “temporal reuse” claim |
| TemporalGS (arXiv 2026) | fixed scene, consecutive frames | temporal geometry/appearance priors; culling + selective tiles | no | approximate adaptive reuse | Hazard to broad “temporal priors/culling/reuse” claim |
| DSD-GS (2026) | dynamic-scene training with static/dynamic decomposition | static-region computation/render caching after decomposition/freeze | yes | frozen subset, not evolving-full-state structural cert | Hazard to “first training-time cache” claim |
| 3DGS-LM (ICCV 2025) | optimization / LM inner solves | cached intermediate gradients for Jacobian-vector products | yes | different object and reuse timescale | Hazard to “first reuse/cache during 3DGS optimization” claim |
| TideGS (2026) | billion-scale training | SSD/CPU/GPU working-set parameter cache and differential streaming | yes | data-movement cache, not raster structure | Hazard to generic training-cache/system claim |
| Kinetic Data Structures (classic) | continuously changing keys | combinatorial sorted structure maintained by adjacent certificates | generic | yes, certificate concept is classic | Direct hazard to claiming certificate-based sorted maintenance itself as novel |

## Safe novelty wording if VRIR0 succeeds
Candidate wording:

> We study fail-closed structural memoization for differentiable 3DGS training: when a discrete training camera is revisited while Gaussian parameters have evolved, we certify that both Gaussian-to-tile membership and tile-local depth order remain exactly compatible with the cached traversal structure before allowing reuse.

This wording must still be re-checked against new publications/patents before submission.

## Forbidden novelty wording
Do not write “first temporal reuse for 3DGS”, “first sorting reuse”, “first cached rasterization in training”, or “first certificate-based sorting”.

## Primary references recorded for DSH refresh
- Neo, ASPLOS 2026, DOI 10.1145/3779212.3790192.
- CaT-GS, CVPR 2026, “Efficient 3DGS Rendering for Large-Scale Scenes with Inter-frame Caching and Tile Scheduling”.
- GSReuse, IEEE TVCG 2026, DOI 10.1109/TVCG.2026.3679889.
- TemporalGS, arXiv:2607.03390.
- Lumina, ISCA 2025 / arXiv:2506.05682.
- DSD-GS, arXiv:2605.30863.
- 3DGS-LM, ICCV 2025.
- TideGS, arXiv:2605.20150.
- CGAL Kinetic Data Structures / Kinetic Sort documentation (classical certificate precedent).
