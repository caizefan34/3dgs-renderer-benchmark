# Upstream facts frozen for P16-RB

These facts were verified before bundle creation.

## Faster-GS — CVPR 2026
Official repository: `nerficg-project/faster-gaussian-splatting`.

Pinned current main:
`117bdbacfd7df6c1f06d9d959b1f5ab02d4d3847`

Pinned current `FasterGSFused`:
`ae665af2f91a4973bb92c40fdc4cba8d12fe3c96`

The official README describes Faster-GS as a highly optimized 3DGS implementation and states that the fused branch fuses backward and optimizer.

## NeRFICG
Pinned:
`e458ac6e41810ad6d9405687c1e698fadd28dac5`

## FastGS — CVPR 2026
Official repository: `fastgs/FastGS`.

Pinned main:
`44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`

The official `train_base.sh` contains scene-specific room/bicycle/garden training commands used by this preregistration.

## Structure-Aware Densification / SADGS — SIGGRAPH 2026
Official repository:
`LinjieLyu/SADGS`

Pinned main:
`029814fb5ed773e4a794cec5115951a64f610ab8`

Official `run_train.sh` uses:
- 3000 iterations for Mip-NeRF360 indoor/outdoor/garden,
- batch size 2 for indoor,
- hybrid optimizer,
- early/warmup structure-aware density control.

The project page reports 2.8× average acceleration vs FastGS-Big with improved LPIPS. This is literature context only; P16-RB measures same-hardware results itself.

## TurboGS
TurboGS (ICML 2026 / arXiv:2606.15924) is relevant literature and reports up to 10× vs vanilla 3DGS on RTX 5090. No official code pin was frozen for P16-RB, so it is not part of this execution matrix.
