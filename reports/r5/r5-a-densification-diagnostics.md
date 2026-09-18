# R5-A: Densification Diagnostics

## Overview

Tracked N_GS_B(t), N_GS_C(t), and R_GS(t) = N_C/N_B at all evaluation checkpoints (500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000) for all 6 paired experiments.

Densification is active from iter 500 to 15000 (config: densify_from_iter=500, densify_until_iter=15000, interval=100). No densification occurs after iter 15000, so N_GS is constant from 15K onward.

## Train Scene

### N_GS(t) and R_GS(t) at all checkpoints

| Iter | B_s0 | C_s0 | R_GS | B_s1 | C_s1 | R_GS | B_s2 | C_s2 | R_GS |
|------|------|------|------|------|------|------|------|------|------|
| 500 | 182,686 | 182,686 | 1.000 | 182,686 | 182,686 | 1.000 | 182,686 | 182,686 | 1.000 |
| 1000 | 191,574 | 191,692 | 1.001 | 193,160 | 193,348 | 1.001 | 193,768 | 191,964 | 0.991 |
| 2000 | 201,371 | 203,880 | 1.012 | 207,136 | 205,471 | 0.992 | 207,509 | 203,116 | 0.979 |
| 5000 | 193,203 | 217,723 | 1.127 | 199,616 | 200,473 | 1.004 | 197,809 | 204,755 | 1.035 |
| 10000 | 270,542 | 298,397 | 1.103 | 283,721 | 282,945 | 0.997 | 263,794 | 293,326 | 1.112 |
| 15000 | 419,478 | 476,167 | 1.135 | 413,455 | 446,052 | 1.079 | 409,375 | 471,798 | 1.152 |
| 30000 | 419,478 | 476,167 | 1.135 | 413,455 | 446,052 | 1.079 | 409,375 | 471,798 | 1.152 |

### Phase Analysis

| Phase | Seed | B Growth | C Growth | R_GS Start→End |
|-------|------|----------|----------|-----------------|
| **0-5K** (early densif.) | 0 | +10,517 | +35,037 | 1.000 → 1.127 |
| | 1 | +16,930 | +17,787 | 1.000 → 1.004 |
| | 2 | +15,123 | +22,069 | 1.000 → 1.035 |
| **5-15K** (active densif.) | 0 | +226,275 | +258,444 | 1.127 → 1.135 |
| | 1 | +213,839 | +245,579 | 1.004 → 1.079 |
| | 2 | +211,566 | +267,043 | 1.035 → 1.152 |
| **>15K** (post-densif.) | all | 0 | 0 | stable |

**Train finding**: Candidate C produces 8-15% more Gaussians than baseline by iter 15K. The divergence starts early (seed 0 already at R_GS=1.127 by 5K). The extra Gaussians do NOT help PSNR for seeds 1 and 2 (both negative ΔPSNR despite higher N_GS), suggesting the skip mechanism causes densification of lower-quality Gaussians.

## Truck Scene

### N_GS(t) and R_GS(t) at all checkpoints

| Iter | B_s0 | C_s0 | R_GS | B_s1 | C_s1 | R_GS | B_s2 | C_s2 | R_GS |
|------|------|------|------|------|------|------|------|------|------|
| 500 | 136,029 | 136,029 | 1.000 | 136,029 | 136,029 | 1.000 | 136,029 | 136,029 | 1.000 |
| 1000 | 137,966 | 138,615 | 1.005 | 139,153 | 138,898 | 0.998 | 138,329 | 139,414 | 1.008 |
| 2000 | 157,792 | 156,387 | 0.991 | 160,961 | 154,031 | 0.957 | 157,327 | 156,467 | 0.995 |
| 5000 | 444,509 | 416,629 | 0.937 | 461,137 | 431,439 | 0.936 | 454,133 | 435,310 | 0.959 |
| 10000 | 856,150 | 883,139 | 1.032 | 879,127 | 885,446 | 1.007 | 858,187 | 903,772 | 1.053 |
| 15000 | 1,074,296 | 1,181,561 | 1.100 | 1,089,627 | 1,194,540 | 1.096 | 1,068,903 | 1,217,421 | 1.139 |
| 30000 | 1,074,296 | 1,181,561 | 1.100 | 1,089,627 | 1,194,540 | 1.096 | 1,068,903 | 1,217,421 | 1.139 |

### Phase Analysis

| Phase | Seed | B Growth | C Growth | R_GS Start→End |
|-------|------|----------|----------|-----------------|
| **0-5K** (early densif.) | 0 | +308,480 | +280,600 | 1.000 → 0.937 |
| | 1 | +325,108 | +295,410 | 1.000 → 0.936 |
| | 2 | +318,104 | +299,281 | 1.000 → 0.959 |
| **5-15K** (active densif.) | 0 | +629,787 | +764,932 | 0.937 → 1.100 |
| | 1 | +628,490 | +763,101 | 0.936 → 1.096 |
| | 2 | +614,770 | +782,111 | 0.959 → 1.139 |
| **>15K** (post-densif.) | all | 0 | 0 | stable |

**Truck finding**: Opposite pattern from train. In early densification (0-5K), Candidate C produces FEWER Gaussians (R_GS ≈ 0.94). But during active densification (5-15K), C overtakes B and ends with 10-14% MORE Gaussians. The crossover happens around iter 5-10K. This pattern is consistent across all 3 seeds.

## Mean R_GS by Phase

| Phase | Train | Truck |
|-------|-------|-------|
| 0-5K (early) | 1.055 | 0.944 |
| 5-15K (active) | 1.122 | 1.112 |
| >15K (final) | 1.122 | 1.112 |

## Interpretation

### Does the skip mechanism affect densification?

**Yes, significantly.** The gradient suppression in Candidate C alters densification dynamics:

1. **Train**: C accumulates more Gaussians early and maintains the excess throughout. The extra Gaussians (R_GS 1.08-1.15) do not improve quality, suggesting they are densified in regions where gradients were suppressed — i.e., "phantom" densification triggered by modified gradient signals.

2. **Truck**: C initially densifies LESS (R_GS 0.94 at 5K) — likely because suppressed gradients fail the densify_grad_threshold in some regions. But then C over-densifies during 5-15K, ending with 10-14% more Gaussians. This is consistent with the skip mechanism destabilizing the densification schedule.

3. **The R_GS pattern is consistent across seeds** for truck (all 3 seeds: 0.936-0.959 at 5K, then 1.096-1.139 at 15K), but varies more for train (seed 1 stays near 1.0 until 10K, while seed 0 jumps to 1.127 at 5K).

### Relationship to ΔPSNR

| Scene | Seed | Final R_GS | ΔPSNR |
|-------|------|------------|-------|
| train | 0 | 1.135 | +0.52 |
| train | 1 | 1.079 | −0.64 |
| train | 2 | 1.152 | −1.16 |
| truck | 0 | 1.100 | +1.94 |
| truck | 1 | 1.096 | −0.00 |
| truck | 2 | 1.139 | +0.81 |

No clear correlation between R_GS and ΔPSNR. The extra Gaussians in C do not systematically improve or hurt quality — the effect on PSNR appears to be seed-dependent noise rather than a systematic consequence of the densification change.

### Conclusion

The skip mechanism perturbs densification (C consistently ends with 8-15% more Gaussians), but this perturbation does NOT produce a reproducible quality improvement. The R4 positive ΔPSNR on T&T was a seed-0 coincidence, not a systematic regularization effect.
