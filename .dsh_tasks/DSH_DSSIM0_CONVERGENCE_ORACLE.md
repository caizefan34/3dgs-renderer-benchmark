# DSH GOAL — DSSIM0_CONVERGENCE_ORACLE

## Mission
Take over and complete the DSSIM0 convergence-oracle lane for 3D Gaussian Splatting training acceleration.

Do NOT brainstorm alternative methods. Execute the existing preregistered experiment, reuse current artifacts, and drive it to a strict PASS / DROP / BLOCKED verdict.

## Local repo
C:\Users\36570\3dgs-renderer-benchmark

## Remote
SSH alias: mx
Remote repo: ~/3dgs-renderer-benchmark
Remote DSSIM0 root: /mnt/storage_pool/liaoyuanjun/dssim0

## Existing assets
- /mnt/storage_pool/liaoyuanjun/dssim0/DSSIM0_PREREG.json
- /tmp/dssim0_continue.py
- /tmp/dssim0_driver.sh
- source checkpoint:
  /mnt/storage_pool/liaoyuanjun/retiregs/RETIRE0_room_source/checkpoints/iter_15000.pt
- source camera sequence:
  /mnt/storage_pool/liaoyuanjun/retiregs/RETIRE0_room_source/camera_sequence.npy

## Current execution state
A room pilot was already launched on GPU6, sequentially:
k=1 -> 2 -> 4 -> 8

Last known driver PID:
785103

Do NOT restart duplicate runs blindly.
First inspect current PID, logs, result.json files, and remote state.

GPU7 is used by RETIRE0. Do not disturb it.

GPU6 contains an existing keeper process. Therefore:
- DSSIM0 room pilot on GPU6 is convergence evidence only.
- timing from this cohort is NON-AUTHORITATIVE.
- never claim speedup from the current GPU6 pilot.

## Baseline loss
Canonical:
L = 0.8 * L1 + 0.2 * DSSIM
every iteration.

## Preregistered schedule
For interval 15K -> 30K:

On DSSIM-active steps:
L = 0.8 * L1 + 0.2 * k * DSSIM

On inactive steps:
L = 0.8 * L1

DSSIM is active once every k steps.

Test:
k = 1, 2, 4, 8

All variants must restore:
- exact same 15K model state
- exact same optimizer state
- exact same camera sequence
- exact same training/evaluation semantics

Do NOT modify the schedule after seeing room results.

## Room quality gate
Reference:
paired k=1 continuation.

For any candidate k:

|delta PSNR| <= 0.15 dB
|delta SSIM| <= 0.005

Promotion condition:
at least k >= 4 must pass.

If k=4 fails:
verdict = DROP_ROOM_CONVERGENCE
STOP this lane.
Do not rescue with k=3, different weights, different start iteration, or post-hoc thresholds.

If k=4 passes:
identify whether k=8 also passes.

## Multi-scene
Only if room passes k>=4.

Repeat equivalent paired protocol on:
- garden
- bicycle

Use topology-stable source checkpoints and exact paired state.

Cross-scene gate:
k>=4 must pass on room, garden, bicycle.

Per scene:
|delta PSNR| <= 0.15 dB
|delta SSIM| <= 0.005

If any scene fails:
verdict = DROP_MULTI_SCENE
STOP.

Report LPIPS if available, but do not invent a post-hoc LPIPS threshold.

## Next stage if multi-scene passes
Only then evaluate whether this can become a real acceleration method.

Two stages:

1. Exact/system optimization of DSSIM implementation.
2. Production implementation of periodic DSSIM scheduling.

For performance work:
- use clean/exclusive target GPU
- no keeper/foreign CUDA contamination
- same cohort
- warmup + repeats
- report median and spread
- measure full E2E, not only loss kernel

The authoritative 9-phase profile established:
garden 30K loss/DSSIM ~= 36.0% E2E
(total ~=69.76 ms, loss ~=25.14 ms).

Preserve this as prior evidence, but do not treat theoretical Amdahl ceiling as measured speedup.

## Performance gate
A production candidate is worth retaining only if:
- quality gates pass
- authoritative E2E speedup is meaningful
- target minimum for production continuation: >=1.25x E2E

If only the loss kernel is faster but E2E is below the gate:
DROP_PERFORMANCE.

## Scientific rules
FAIL-CLOSED.

Forbidden:
- lowering thresholds
- changing loss weights post hoc
- inventing rescue variants
- mixing hardware/software cohorts
- calling GPU6 pilot timing authoritative
- claiming speedup from theoretical ceiling
- reporting kernel-only speedup as training speedup
- hiding failed scenes
- silently changing reference_v1 semantics

## Required deliverables
Write under:
C:\Users\36570\3dgs-renderer-benchmark\_dssim0_results\

Mirror to:
 /mnt/storage_pool/liaoyuanjun/dssim0/final/

Required:
1. DSSIM0_SOURCE_AUDIT.json
2. DSSIM0_ROOM_ORACLE.json
3. DSSIM0_MULTISCENE_ORACLE.json
4. DSSIM0_PROFILE_ANALYSIS.json
5. DSSIM1_IMPLEMENTATION_CORRECTNESS.json
6. DSSIM1_AUTHORITATIVE_TIMING.json
7. DSSIM0_MASTER_VERDICT.json
8. DSSIM0_MASTER_REPORT.md
9. raw scripts/logs needed for reproduction
10. MANIFEST.sha256

Final verdict must be exactly one of:
- PASS_METHOD_CANDIDATE
- DROP_ROOM_CONVERGENCE
- DROP_MULTI_SCENE
- DROP_HEADROOM
- DROP_CORRECTNESS
- DROP_PERFORMANCE
- BLOCKED_SOURCE_INTEGRITY
- PARTIAL_BLOCKED_BY_GPU

Report sections:
OBSERVED
EVIDENCE
INFERENCE
NOT_MEASURED
CONCLUSION

Do not update the KB with a positive conclusion until the master verdict is frozen.
