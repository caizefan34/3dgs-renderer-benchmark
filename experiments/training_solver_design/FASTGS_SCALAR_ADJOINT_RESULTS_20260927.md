# FastGS scalar-adjoint bounded result (2026-09-27)

This tests the frozen engineering gate in `FASTGS_SCALAR_ADJOINT_GATE_20260927.md`. It is not a novel 3DGS method, a complete-training speed result, or an image-quality result. The author FastGS source remains unchanged; the candidate lives only in `fastgs_scalar_adjoint_trial/cuda_rasterizer/backward.cu` and its isolated EPIC-05 extension prefix.

## Correctness gate passed

On four fixed room@4100 **training** images at the author's effective width 1600 and original `0.8 L1 + 0.2(1-fusedSSIM)` loss, forward RGB and loss matched the native extension exactly. All six parameter-gradient groups and the screen-space gradient passed the frozen relative-L2 ≤1e-3 or max-absolute ≤1e-7 gate; the largest reported relative-L2 difference was about 9.20e-6. This is only fixed-state numerical parity, not later-stage, boundary-case, topology-trajectory or held-out quality parity. Remote raw tensors: `/root/fastgs_scalar_probe_20260927/{native,candidate}.pt`.

## Whole-block timing gate failed

The two completed alternating pairs used the same frozen room model and four cameras, 16 warmup steps per process, then five blocks of 128 full-image forward/original-loss/backward steps. The median wall seconds per block and paired candidate/native ratios were:

| Pair | Native | Candidate | Ratio | CUDA-event ratio |
|---|---:|---:|---:|---:|
| 1, native→candidate | 0.756526087 | 0.728124625 | 0.962458 | 0.962451 |
| 2, candidate→native | 0.754606483 | 0.730644391 | 0.968246 | 0.968244 |

The frozen promotion rule required the median of **three** paired wall ratios ≤0.95 and all three pairs faster. With two observed ratios above 0.95, the best possible three-pair median is the smaller observed ratio, 0.962458. Therefore pair 3 cannot make the candidate pass; the experiment stopped early without changing the rule. The observed frozen-block gain is about 3–4%, not the required ≥5% gate. Do not extrapolate it to complete training wall time. No garden test or complete training was started.

Timing raw logs on EPIC-05: `/root/fastgs_scalar_probe_20260927/timing_{native,candidate}_{1,2}.log`; SHA256 candidate 1 `cdb0691275cafffeac60b142649c4daa83cb28d995fbbda761a8058184529563`, candidate 2 `2e82242ce60511990354d831b5ff6482eacef431b6a2d6dd61693719487742a4`, native 1 `ce1191b20f145423c9c67e7f935d496f129fbaa9dcd6445e4e4097875a11e3bf`, native 2 `2f7a7e19a54c38290432e47c503740084337399adeefd51f0705678778e2acc5`. Candidate extension binary SHA256 `88f1335d36281d43a0d682958ca36a2d8080d263606c7c49c091dfa671ea30d9`. The four logs identify the same PLY hash `64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc` and probe hash `4d00e5e143a6b330a31b40b4c39aec76a08986169bde21b07f6ae194fc31b460`.

Decision: stop this scalar-adjoint version at the preregistered timing gate. It demonstrates an algebraically equivalent backward component and a modest local speed opportunity, but no new training method or progress toward the required multiscene, multiseed PSNR/SSIM/LPIPS and complete-wall claim.
