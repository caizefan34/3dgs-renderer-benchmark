# GSC0 P0 — Gaussian State Compression

Status: P0 ORACLE ONLY. NO CUDA. NO TRAINING.

## Research question

Does mature 3DGS training contain Gaussian-level state redundancy that allows adaptive precision reduction without changing optimization behavior?

The hypothesis is NOT generic mixed precision. It is:

Gaussian-level adaptive numerical representation:

    Gaussian_i + state history + visibility
              |
              v
       precision policy

The unit of decision is a Gaussian primitive, not a whole tensor.

## Hard execution rule

LOCAL WINDOWS MACHINE = CONTROL PLANE ONLY.

Allowed locally:
- read/edit files
- git operations
- package transfer
- inspect text/json outputs

Forbidden locally:
- training
- rendering
- CUDA build/JIT
- GPU benchmark
- ncu/nsys
- scene-scale trace replay
- large PyTorch/NumPy experiments

All experiments MUST execute on EPIC-05.

## P0 objective

Do not implement compression.

First answer:

1. How much of each Gaussian attribute becomes stable?
2. Which attributes are stable?
3. What error would reduced precision introduce?
4. Is there enough safe compression headroom?

Possible verdicts:
- KEEP_FOR_GSC1
- DROP_NO_STABLE_STATE
- DROP_UNSAFE_ERROR_BOUND
- DROP_UNECONOMIC
- BLOCKED
