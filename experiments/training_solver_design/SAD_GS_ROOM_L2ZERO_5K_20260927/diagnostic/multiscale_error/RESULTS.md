# Room residual-scale attribution (2026-09-27)

This is a read-only diagnostic on the frozen 39 test-render PNGs of original-loss SAD-GS 5K and timed from-zero FastGS-Big 30K, with byte-identical GTs. It does not train or modify either method. Each RGB residual image is orthogonally projected onto nonoverlapping piecewise-constant blocks of size 4, 16 and 64 pixels; partial boundary blocks use their actual pixel counts. Telescoping projected squared energies gives `<4px`, `4–16px`, `16–64px`, and `>64px` block-scale components whose sum equals the full pixel MSE. These are **block scales**, not strict Fourier bands or a source attribution.

| Mean MSE difference, SAD minus FastGS | All 39 official test views | Descriptive 38 views excluding `00021.png` |
| --- | ---: | ---: |
| Total | .000358437 | .000119660 |
| `<4px` | -.000005962 | -.000003873 |
| `4–16px` | +.000021653 | +.000013674 |
| `16–64px` | +.000057966 | +.000032971 |
| `>64px` | +.000284779 | +.000076888 |
| Share of total excess MSE from `>64px` | 79.45% | 64.26% |

SAD has more `>64px` residual energy on 35/39 individual views and less `<4px` energy on 23/39. The one unusually bad view has a large effect on mean MSE, but even the **descriptive** 38-view comparison assigns most excess energy to large block scales. The official 39-view PSNR/SSIM/LPIPS aggregates must retain `00021.png`; average MSE and its band shares are not mean PSNR. The separate per-view test-GT-fitted RGB-bias oracle did not narrow the mean PSNR gap, so a spatially constant color shift is not sufficient either.

This evidence deprioritizes extra fine-detail/frequency splitting as a response to the current room PSNR deficit. Broad residuals can come from spatially varying appearance, coverage, opacity, occlusion or geometry, and do not identify a specific 3D parameter or a removable dominant training cost. There is still no credible original quality-and-cost mechanism or reason to start another full training arm.

`analyze.py` and `result.json` were copied byte-for-byte from the CPU-only EPIC-05 run. SHA256: script `fa30faec97699a07a6b89dc505d4e8b6939c086ae4090c141454afb50d85e56e`; result `7d66f31b844760330dad2556045f7f89098553385e7028cd6edecd6a3971dbe2`. No GPU process was launched; GPU3 remained idle.
