# Trainable HiGS Forward Architecture Map

Source baseline: `77ab983ffe43420b2131669cb35776b883ca4c3c`. This is an engineering map only; no macro-tile training implementation was made.

| Stage | Official HiGS inference | Current Trainable B2 | Can reuse directly? | Missing training state |
| --- | --- | --- | --- | --- |
| F0/I0 | Inference state owns a visible bitmask | HiGS cull produces visible subset IDs/radii | No | Representation and master-to-visible mapping differ |
| F1 | No equivalent gather stage | Gather visible FP32 masters | Keep | None for the forward-only boundary |
| F2/I1 | Projection emits FP32 means2d/depths, FP16 conics | `fully_fused_projection` emits B2 FP32 projected state | Not directly | Inference conic precision differs |
| F3/I1 | Fused/generic SH emits FP16 colors | B2 evaluates FP32 colors | Not directly | Inference color precision differs |
| F4/I2-I4 | Count, chunk bases, macro offsets, fill, segmented macro sort | Conventional `isect_tiles` plus global radix sort | Replace | Macro offsets/sorted IDs must be retained |
| F5/I5-I6 | 1024 Gaussian batches, 32-Gaussian masks, macro raster, post-blend | Conventional pixel raster | Replace | Last ID and termination state are not emitted |
| F6 | No autograd save | Saves forward tensors for native backward | Replace for prototype capture only | Explicit retained macro state |

Official inference uses 8x4 macro tiles (32 fine tiles), 1024-Gaussian macro batches, and 32-Gaussian mini-batches. `WarpBitTranspose` converts per-Gaussian 32-bit fine-tile overlap masks into per-fine-tile Gaussian masks. Raster output is FP16 RGBT, where the fourth channel is transmittance.

The exact B2 training graph is:

`F0 cull → F1 gather → F2 fully_fused_projection → F3 SH → F4 conventional isect/sort → F5 conventional raster → F6 autograd save`.

The official inference graph is:

`projection+SH → macro count → chunk bases+offsets → fill → segmented macro sort → 1024-Gaussian macro batches → 32-Gaussian mini-batches/WarpBitTranspose/fine masks → macro raster → post blend`.

A minimum forward-only prototype leaves B2 F0-F3 untouched and replaces only F4-F5 with the official macro stages. It returns RGB plus alpha/transmittance and captures validation-only macro offsets, sorted IDs, batch offsets, active masks and final transmittance. It must compare RGB, alpha, support/visibility, semantically relevant order and last contributing Gaussian against the same B2 forward input.

The present inference implementation cannot establish FP32-equivalent training output on its own: conics, colors, partial RGBT and final RGBT are FP16. It also does not expose per-pixel last Gaussian IDs or per-pixel/batch early-termination information. Those are engineering blockers for an exact backward contract, not proposed solutions.

Full inventory, lifetimes, backward-state status, topology invalidation and prototype boundary are in `artifacts/trainable-higs-forward-map/`.

