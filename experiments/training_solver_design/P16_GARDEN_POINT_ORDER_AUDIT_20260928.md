# P16 garden point-cloud record-order audit

2026-09-28. Initial read-only diagnosis of the P16 garden failure, followed by an isolated conversion. **Superseded by the later official-source recovery below.** The earlier conversion explains one byte-layout symptom but is not a valid dataset repair or baseline input.

## Authoritative later recovery and corrected baseline state

After this initial audit, the P16 operator independently traced the defect to a September 16 garden-only rebuild and approved a full garden heal and rerun. The authoritative remote record is `/root/epic05_p16rb/artifacts/P16_DATA_HEAL_garden.json`; the former corrupted runs are retained under `runs/<method>/garden_flatdata_corrupt/`. The live P16 data now differ from the old hashes below.

The corrupt source had a flat 1,839,236-point gray cloud with zero tracks, and `images.bin` had its 2D keypoints stripped (15,733 bytes versus 47,011,789 bytes in the official archive). Merely reordering its point records would preserve the wrong synthetic cloud and missing image observations. The operator recovered `cameras.bin`, `images.bin`, and `points3D.bin` from the official `360_v2.zip`, checked archive CRC/size, image names, camera poses, and one image's byte identity, then regenerated the PLY through the FastGS loader path. The repaired official sparse model loads with 185 images and **138,766** points; its source `points3D.bin` SHA256 is `be52e343ad059bbfce00e2b22d397de75dad2d6f1ac7cf359b1b926c2b0482c8`, and regenerated PLY SHA256 is `3904ceabc4a1630bb0bac21fb2be5f2620dd1009b6830375e40d33de9f629b8c`.

All five garden methods were rerun from initialization on the repaired data. Their common-evaluator 1920×1080 garden means and recorded native training timers are:

| Method | PSNR | SSIM | LPIPS | Native training timer, s |
|---|---:|---:|---:|---:|
| FastGS | 26.699 | .819 | .224 | 267.8 |
| FasterGS main | 26.859 | .842 | .156 | 485.7 |
| FasterGS fused | 26.849 | .842 | .157 | 528.2 |
| SADGS | 26.647 | .843 | .144 | 105.8 |
| P12_C0 | 28.451 | .866 | .175 | 778.8 |

Root independently read all five `metrics_common.json` files after recovery: each has `status=SUCCESS`, 24/24 expected test images, resolution `1920×1080`, exactly the same set of image stems, and byte-identical GT hash manifests across methods.

These are **existing-method, one-seed garden baselines**, not an original method or a winner of the user's target. The recorded native timer boundaries differ: Graphdeco `Training time`, NeRFICG training callbacks, and P12 summed loop times; use a common complete-process boundary before claiming wall-time rankings. Native author recipes also do not all use the same per-step original loss, so this P16 table is a strong-baseline census, not the final same-loss comparison. The remote final chain has passed its operator gate and is performing curves/profiles; do not restart it or reuse GPU 2 concurrently. The originally proposed point-order conversion is invalid for any rerun.

Earlier EPIC-05 garden diagnostics trained after the corrupt September 16 rebuild also need provenance review. In particular the garden@4100 model in `FASTGS_CAPACITY_EXCHANGE_SCREEN.md` has 4,111,216 points and PLY SHA256 `d396092ca139fa0e6aea4d17a085ca0239bf2ff0ec2bbbe0ee3c5ab539feb5fb`; the garden@4100/30K states in `fastgs_residual_cost_overlap_20260927/REPORT.md` and the garden rows in `FASTGS_STAGE_PROFILE_RESULTS.md` were collected from that shared garden path on September 27. Their garden-specific quality and cost inferences should be quarantined until rerun or source-image/initialization hashes show otherwise. Room results and any independently sourced garden experiments are not invalidated by association.

P11's `REAL_TOPOLOGY_RESULTS.md` garden@2000 checkpoint already has 1,841,621 Gaussians, close to the corrupt 1,839,236-point initialization; its EPIC-05 September-26 garden comparisons also need provenance review. The local Windows `data/datasets/mipnerf360/garden/sparse/0` is a separate, official-size copy: `points3D.bin` is 14,835,394 bytes and `images.bin` is 47,011,789 bytes, matching the repaired official SHA256 values `be52e343...` and `9490fc63...`. This local copy should not be conflated with the old remote corrupt input.

## Finding

The source garden `points3D.bin` is not in COLMAP's required `point3D_id, xyz, rgb, error, track` record order. It stores `xyz, point3D_id, rgb, error, track`. Its first record starts with xyz `(4.9963812828, 1.7507487535, -0.9372210503)`, followed by ID `1`. A standard parser instead treats the first x double's bits as the ID, takes `(1.7507487535, -0.9372210503, 5e-324)` as xyz, and produces the observed zero-z PLY. The original garden PLY's 1,839,236 z coordinates are all exactly zero. Room and bicycle PLYs do not show this collapse.

The source binary SHA256 is `2aada8baa1418ab6d87f448240b92e9acaf47f59ffec20482da41377d0434555`; the source/common PLY SHA256 is `2b053e7ef498e8763d5a38dedf12e082a7e7311eb6097323c0e7add20c815d6c`. The P16 common binary SHA256 is `4ebf360816733df10c0a37e5e7dc4abcb97387575acb8e1648675a0ce96cb625`. The existing `p16_fix_garden_colmap.py` assumes standard record order and renames apparent duplicate IDs; those IDs are actually x-coordinate bit patterns, so it cannot repair the true layout.

The P16 common evaluator uses the same garden GT folder across methods and hashes GT pixels to map index-named renders. Yet the staged native garden PSNRs are FastGS `12.687`, FasterGS main `12.996`, FasterGS fused `12.615`, SADGS `13.225`; P12_C0 is `22.909`. The four low results do not by themselves prove that this point-cloud defect is the sole quality cause, but the native baselines are not interpretable as valid same-protocol garden quality comparisons until the input and reruns are validated. A process `clean=true` and successful metric script do not certify semantic input integrity.

## Isolated correction and checks

`p16_rb/scripts/p16_audit_garden_point_order.py` reads the **original source** binary and writes a standard-order `points3D.bin` plus correctly positioned `points3D.ply` under `/root/epic05_p16rb/audit_garden_corrected_20260928/sparse/0`. It also copies the P16 camera/image metadata there. The converter passed a one-record synthetic byte-layout and PLY-coordinate test. On the full source it found 1,839,236 monotone IDs from 1 to 1,839,236, no nonfinite records, zero tracks, xyz minima `(-44.790844, -34.003365, -23.989439)` and maxima `(56.925411, 9.191805, 35.236153)`. The staged binary and PLY SHA256 values are `441f2684b34e1513c385d450cf78bec176c5e480d34a7bd0c020ea180a19f123` and `b1c4f39613174ad149fd16a540e98676f4cddab4b98924967efd5acf0ba09ea4`.

Pinned P16 FasterGS environment `pycolmap.Reconstruction` loads the staged model as 1 camera, 185 images and 1,839,236 points. The staged PLY has 1,839,236 nonzero z values, spanning `-23.989439` to `35.236153`. This checks format and coordinate preservation, **not** reconstruction quality or whether the source's untracked, constant-gray points are a scientifically appropriate initialization.

## Original decision before the operator's independent recovery (historical)

The initial recommendation was to hold the operator review gate and rerun all garden baselines with verified official data. That recovery and rerun were subsequently performed by the P16 operator, as documented above. The isolated 1,839,236-point conversion remains a diagnostic artifact only and must not be substituted for the official 138,766-point reconstruction.
