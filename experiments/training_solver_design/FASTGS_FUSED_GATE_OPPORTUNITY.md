# Continuous split gate: real pixel-work and isolated VJP cost gate

2026-09-27. This is a cost opportunity check for a prospective full-resolution
FastGS split-probation method, **not** a method with measured training benefit.
The author renderer/trainer was not changed.

## Real room split event

The unchanged author FastGS commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`
and the recorded room@4100 pre-event PLY (`64160970…cc48f85bc`) select 9,345
native split parents. Four training cameras at 1600×1066 were evaluated on
GPU 2 of EPIC-05. The author CUDA renderer supplied projected Gaussian rows
and physically depth-sorted tile lists. The diagnostic scalar kernel applied
the same raw-opacity cutoff and early-stop rule as the earlier endpoint/VJP
reference; it counted every pixel–tile-list row it visited, every row above
the raw cutoff, and every accepted nonzero-opacity row.

| Train view | Union/native-post tile pairs | Half-gate/native-post visited rows | Half-gate/native-post active rows | Shadow share of raw-above-cutoff rows |
| --- | ---: | ---: | ---: | ---: |
| DSCF4668 | 1.0243× | 1.0243× | 1.0234× | 4.55% |
| DSCF4770 | 1.0166× | 1.0162× | 1.0194× | 3.87% |
| DSCF4873 | 1.0163× | 1.0159× | 1.0185× | 3.53% |
| DSCF4976 | 1.0098× | 1.0099× | 1.0098× | 1.83% |

The half-gate shadow-support list therefore adds 0.99–2.43% scalar visited
rows over native post-split in this local sample. There are 1.44–3.93 million
shadow-family raw-above-cutoff interactions per view. As an internal endpoint
check, the count of accepted **nonzero-alpha** interactions at gate=0 equals
the native pre-split count exactly, and at gate=1 equals native post-split
exactly, on all four views. This checks the diagnostic's support accounting;
it does not establish training gradient parity.

## Gate arithmetic versus atomic accumulation

A second diagnostic used the same union buffers at gate=.5 and alternated
15 CUDA-event samples per mode. RGB output was bitwise identical between
modes; the no-atomic and atomic modes produced the same debug-pixel VJP.

| Train view | RGB only | RGB + VJP math, no atomics | RGB + VJP math + family atomics | Atomic increment |
| --- | ---: | ---: | ---: | ---: |
| DSCF4668 | 2.3098 ms | 4.4589 ms | 4.4792 ms | 0.0204 ms |
| DSCF4770 | 2.0969 ms | 3.9427 ms | 3.9607 ms | 0.0180 ms |
| DSCF4873 | 1.2383 ms | 2.3282 ms | 2.4144 ms | 0.0863 ms |
| DSCF4976 | 1.0839 ms | 2.0411 ms | 2.1479 ms | 0.1068 ms |

The extra **second full pixel traversal and VJP arithmetic** is the dominant
cost in this isolated kernel; its atomic writes add only 0.018–0.107 ms here.
The earlier separately measured gate pass cost 3.7–4.4 ms per view on a
particular camera/repeat protocol, so an additional standalone pass per
probation step remains unattractive. A fused native backward might reuse its
existing traversal and avoid that duplicated pass. The two diagnostic kernels
have different operations and scheduling from the author rasterizer, so these
numbers are **not** a predicted fused-trainer overhead or speedup.

## Pinned renderer integration point

Source inspection of the pinned checkout narrows the required CUDA change.
`cuda_rasterizer/forward.cu` builds each pixel's compositing state at the
raw-opacity cutoff (`renderCUDA`, around lines 384–423). The production
backward called by `BACKWARD::render` is `PerGaussianRenderCUDA` in
`cuda_rasterizer/backward.cu`, around lines 408–613: it uses bucketed,
Gaussian-major tile traversal and already forms a per-pixel `dL/dalpha`
before reducing parameter gradients per Gaussian. It is **not** the simple
pixel-major second pass used by our diagnostic. The next prototype must pass
family IDs, roles and gate values through the rasterizer interface; use
**ungated** opacity for tile support; and apply the gated alpha only after
raw cutoff in forward and backward. For raw pixel alpha `a` and family gate
`g`, the candidate chain rule is:

```text
parent: alpha = 1-(1-a)^(1-g)
        d alpha/dg = (1-alpha) log(1-a)
        d alpha/da = (1-g) (1-a)^(-g)
child:  alpha = 1-(1-a)^g
        d alpha/dg = -(1-alpha) log(1-a)
        d alpha/da = g (1-a)^(g-1)
```

The bucketed backward could add `dL/dalpha * d alpha/dg` to a per-Gaussian
register sum, then scatter once per tile-Gaussian to its family; all native
parameter gradients must multiply by `d alpha/da`. These are implementation
requirements derived from the current source and chain rule, **not** a CUDA
parity or wall-time result. The author's clamp at alpha=.99 and early stop
must be handled in the same semantics as its existing surrogate gradients.

## Decision and next gate

This cost evidence reopens a narrow **engineering opportunity** for a fused
continuous gate; it does not reopen the weak one-shot veto or child-choice
policies as strong quality mechanisms. A credible implementation must retain
ungated candidate tile support, reproduce native images at both endpoints,
pass pixel/family VJPs under physical depth interleaving, and time full
forward/backward/Adam with gates in the author trainer. Parent and children
must coexist only for a bounded probation window, with optimizer-state
transfer and commit/rollback checked. Any quality test must use ordinary
full-resolution training views, include the gate work in complete process
wall, and confirm on untouched scenes/seeds. If the fused variant's true
overhead cannot be repaid by later point/work reduction or a substantial
quality gain, stop this direction.

Reproduction: `fastgs_shadow_work_cuda.cu`, `fastgs_shadow_work_probe.py`,
`fastgs_shadow_work_result.json`, `fastgs_shadow_work_analysis.py/json`,
`fastgs_event_gate_cost_cuda.cu`, `fastgs_event_gate_cost_probe.py`, and
`fastgs_event_gate_cost_result.json` in this directory. Executed CUDA source
SHA256: work counter `d899b1dc18037579ab5a4cdfef4c7a7952a4c7e39f788ef9995c569116dd0d`;
cost toggle `e81ac1533eb758536ec02829d4ad8478a991a1a12675a814e84bce78392cf031`.
