# Room fixed-output RGB bias oracle (2026-09-27)

This is a read-only error attribution test, **not a deployable correction, training method or official metric**. The frozen SAD-GS original-loss 5K render and the timed from-zero FastGS-Big 30K render were compared on the same 39 room test images at 1600x1066. All 39 GT PNG files match byte-for-byte between systems. For each method and each view independently, the script fits the three-channel constant addition `mean(GT-render)` using that test view's GT, clips the corrected RGB to `[0,1]`, and recomputes PSNR. This is an optimistic oracle for a spatially constant additive color correction.

| 39-view mean PSNR dB | Original PNGs | Test-GT-fitted RGB-bias oracle | Oracle gain |
| --- | ---: | ---: | ---: |
| SAD-GS 5K original-loss | 31.218957 | 31.403328 | +.184371 |
| FastGS-Big timed from-zero 30K | 32.205460 | 32.399964 | +.194504 |
| FastGS minus SAD gap | .986503 | .996636 | +.010133 |

The recomputed uncorrected means agree with the retained author metric JSONs to within `3.1e-6` dB for SAD and `1.6e-7` dB for FastGS. The largest SAD oracle gain is on the known `00021.png` veil view: +1.197441 dB, with fitted RGB additions `[-.065212, -.048445, -.025053]`; that view remains at only 21.051747 dB versus FastGS's original 30.413367 dB. Across all 39 views, SAD's mean oracle gain is slightly **smaller** than FastGS's. Thus a per-view global RGB offset cannot close the strong-baseline PSNR gap, even with privileged test GT fitting. This does not rule out spatially varying errors, geometry/visibility effects or richer appearance models, and it gives no training-wall cost mechanism or novelty claim. Stop the simple exposure/color-offset repair direction.

The CPU-only script used the existing PNGs and no GPU, training, author-code edits or new renders. It wrote `result.json` with all per-view MSE, fitted RGB biases and PSNRs. SHA256: `analyze.py` `3993552b3135a70219ad19509a81c40dea2db793aad78e228d2d989afeeedca7`; `result.json` `298891f15e9c2b7b3e6f8173daee9d40b621fff772e4da0e17bcd5fd5ce09bec`. Local copies match EPIC-05 byte-for-byte.
