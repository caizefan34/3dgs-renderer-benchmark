# Tile-prefix transmittance gate: results

Date: 2026-09-28. Outcome: **STOP at the ideal-oracle gate.** The frozen native-list upper bound was 11.5468%, below the required 40%, and no sampled camera reached the 20% per-camera threshold. The certified bound was therefore not computed, as required by the frozen protocol.

## Frozen inputs and execution

- Author checkout HEAD: `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`.
- FastGS-Big room iteration-30000 PLY SHA256: `995aa0e0dd525c9d3728a533d6bec3465d340dd3791e36253c93c741b2cc80df` (572,546 Gaussian vertices).
- Checkpoint `cfg_args` SHA256: `8d8930a36bfda7c8f2d9ea5521483962b6188f9a97321a224eeccd05864387dd`; it records `eval=True`, `resolution=-1`, `sh_degree=3`, and `white_background=False`.
- Core source SHA256: `rasterizer_impl.cu` `72b87a59f1fdbe50eb547b81d681465d7e1164ad5c48f704a216fd1b4a59bb7b`; `forward.cu` `c3604b5c1b8b1632ca1f2f6238c4dc2eec5bac50958c27c40bb27b05cd9aa14f`; `rasterize_points.cu` `da1ba6fcf4c5860cee7fa92e9fa2a548052e838af83c587874a854d3bc45a7f3`.
- Frozen camera/tile manifest SHA256: `f4ca645b5c7152c37a88c74873081c74f58042c037a9ca1b975c4a7b4bbc1a4a`. It selects training indices `0,39,77,116,155,194,232,271`, corresponding to `DSCF4668`, `DSCF4712`, `DSCF4756`, `DSCF4800`, `DSCF4845`, `DSCF4889`, `DSCF4933`, `DSCF4977`. Each image uses its native `100×67` tile grid and 256 tiles sampled by the frozen midpoint rule, for 2,048 tiles total.
- The author `eval=True` split has 311 named images: indices divisible by 8 in sorted `image_name` order are held out, leaving 272 training cameras. Only names needed to apply that split were read for all images; poses were unpacked only for the selected training cameras. No image/GT files or held-out poses were read.
- The selected-camera renders used the unchanged author projection path at 1600×1066, `mult=0.5`, black background, and the original native rasterizer. GPU3 was an idle A100 80GB at launch. Environment: Python 3.10.21, PyTorch 2.9.1+cu128, NumPy 2.2.6, CUDA 12.8.

## Native export and parity

The autograd node exposed the original `geomBuffer`, `binningBuffer`, and `imgBuffer`. The script decoded `point_list` and sorted `point_list_keys` from the native binning chunk using the source's 128-byte alignment and `ctx.num_rendered`; tile ranges were reconstructed from the high 32 key bits using the exact `identifyTileRanges` grouping rule. Projected centers, conic/opacity, and native SH-converted RGB came from the native geometry buffer. Native `final_T` and `n_contrib` came from the image buffer. The author sources and exact chunk layouts are recorded in the manifest and script.

CPU/native parity was checked on sampled tile `(tx=3, ty=2)`, tile ID 203, in each selected camera (256 pixels each). Across all eight tiles: **zero `n_contrib` mismatches, zero RGB errors above 1e-5, and zero final-T errors above 1e-6**. Maximum absolute RGB error was `3.1293e-7`; maximum absolute final-T error was `1.1921e-7`. Every mismatch list is empty in `ideal_gate_results.json`.

## Ideal-oracle result

The CPU replay used the native depth order, `power>0` skip, `alpha=min(0.99, opacity*exp(power))`, `alpha<1/255` rejection, and `test_T<1e-4` early stop. It retained the entry that triggers each pixel's done flag and counted only entries strictly after the largest per-pixel trigger prefix. A tile with any pixel never marked done contributes zero removable entries.

| Training index | Camera | Original entries | Ideal removable | Fraction |
|---:|---|---:|---:|---:|
| 0 | DSCF4668 | 35,658 | 3,619 | 10.149% |
| 39 | DSCF4712 | 40,880 | 7,367 | 18.021% |
| 77 | DSCF4756 | 27,520 | 3,271 | 11.886% |
| 116 | DSCF4800 | 33,375 | 3,057 | 9.160% |
| 155 | DSCF4845 | 42,346 | 4,982 | 11.765% |
| 194 | DSCF4889 | 17,379 | 907 | 5.219% |
| 232 | DSCF4933 | 14,512 | 1,127 | 7.766% |
| 271 | DSCF4977 | 28,890 | 3,447 | 11.931% |
| **Pooled** | 8 cameras | **240,560** | **27,777** | **11.5468%** |

No camera reached 20% (required: at least four). There were no empty selected tiles. Of 2,048 tiles, 960 had every pixel eventually marked done; 937 had a non-empty removable suffix, while 23 reached completion only after the final list entry. The ideal gate fails both frozen thresholds, so the experiment stopped here. No depth-bin certificate, bound-evaluation count, certificate storage count, training, held-out render, quality metric, or speed claim was produced.

## Artifacts and limits

- `run_gate.py` SHA256: `3a2adcef3a8d35adf7673e21192b51784eee174df6b3848c405650784970d52a`.
- `camera_tile_manifest.json`: exact frozen camera/tile IDs, selected pose/intrinsic parameters, source hashes, and model hash.
- `ideal_gate_results.json`: raw per-camera and per-tile counts, parity results, and pooled decision.
- `native_tiles_*.npz`: exact native list IDs/keys and projected Gaussian data for each selected tile, plus native `n_contrib` and final transmittance values. These are the offline replay inputs.
- `COMMANDS.txt`: the two executed commands (manifest freeze, then ideal gate).

The result is an upper-bound no-go for this one frozen FastGS-Big room checkpoint and sample. It does not establish general impossibility, originality, or a wall-time effect. Removed suffix entries already lie beyond native pixel early termination and are not counted as saved accepted-pixel backward work.
