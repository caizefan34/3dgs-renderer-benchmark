# ReproTopo0 — Deterministic Topology State Machine (FROZEN)

Packet: REPROTOPO0_PACKET_20260930  
Generated: 2026-09-30T14:27:33Z  
Execution: EPIC-05 A100 cluster (8x A100-SXM4-80GB)  
Local GPU used: False  

## Mechanical verdict

```json
{
  "verdict": "REPROTOPO0_STRONG_PROMOTE",
  "strong": true,
  "promote": true
}
```

## Binary / checkpoint identity

- C0/core/fused_ssim hashes match packet pins: **True**
- Authoritative ckpt hashes match: **True**

## Policy parity (Sec.4)

policy_parity_pass = **True**

| scene | iter | N_in | orig_N_out | rt_N_out | clone o/r | split o/r | pruned o/r | pass |
|---|---|---|---|---|---|---|---|---|
| room | 10100 | 710652 | 716054 | 716054 | 5749/5749 | 635/635 | 982/982 | True |
| room | 10200 | 716054 | 721657 | 721657 | 6010/6010 | 680/680 | 1087/1087 | True |
| room | 10300 | 721657 | 732069 | 732069 | 10327/10327 | 1063/1063 | 978/978 | True |
| room | 10400 | 732069 | 737626 | 737626 | 5870/5870 | 605/605 | 918/918 | True |
| room | 10500 | 737626 | 742746 | 742746 | 5586/5586 | 594/594 | 1060/1060 | True |
| bicycle | 10100 | 2960907 | 2993082 | 2993082 | 29197/29197 | 5552/5552 | 2574/2574 | True |
| bicycle | 10200 | 2993082 | 3019257 | 3019257 | 24265/24265 | 4332/4332 | 2422/2422 | True |
| bicycle | 10300 | 3019257 | 3050778 | 3050778 | 28500/28500 | 5332/5332 | 2311/2311 | True |
| bicycle | 10400 | 3050778 | 3079387 | 3079387 | 26665/26665 | 4238/4238 | 2294/2294 | True |
| bicycle | 10500 | 3079387 | 3109418 | 3109418 | 27582/27582 | 4817/4817 | 2368/2368 | True |

## Determinism stress (Sec.5)

- same-GPU (64x) deterministic: **True**
- cross-A100 (GPU0-3) identical: **True**
- permutation (8x) invariant: **True**
- exact-tie stress deterministic: **True**
- total NaN/Inf: 0 ; total ID collisions: 0

## Baseline audit (Sec.2)

- baseline already full-deterministic (same-GPU + cross-A100): **True**
- original permutation-invariant: **False**
- candidate adds stronger property (permutation/tie invariance): **True**

## Timing (Sec.6)

- max event wall ratio (reprotopo/original): **0.8087**
- amortized full-training overhead fraction: **0.000000**
- extra peak VRAM fraction: **0.0471**

## Boundary

This is NOT a new density-control algorithm. It makes topology-changing 3DGS training a reproducible state machine once deterministic gradients are available. The captured trace is workload-only, not canonical continuation evidence.
