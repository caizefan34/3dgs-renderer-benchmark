# P13-X X1/X2 integration — exact renderer meta, no CUDA modification

## Key fact
At frozen gsplat commit `512d366b...`, `rasterization()` already returns exact renderer intermediates in `meta`: `radii`, `depths`, `isect_ids`, `flatten_ids`, `isect_offsets`, `tile_width`, `tile_height`. Therefore X1/X2 do **not** require a debug CUDA build.

## Integration rule
Copy `/root/epic05_p12/publication_trainer_p12.py` to `/root/epic05_p13x/publication_trainer_p13x_probe.py`. Do not edit the P12 file.

At diagnostic iterations `15000,15250,...,30000`, render the 8 preregistered true COLMAP anchor cameras under `torch.no_grad()` using the *same C0 rasterization call and arguments* as training. Diagnostics must run outside CUDA-event timing and must not call optimizer/densification.

Immediately after each diagnostic rasterization returns `(render, alpha, meta)`, call:

```python
from p13x_probe_core import snapshot_from_meta
snapshot_from_meta(
    meta,
    iteration=iteration,
    scene=args.scene,
    anchor_id=anchor_id,
    out_dir='/root/epic05_p13x/artifacts/x12_snapshots',
)
```

### RNG non-contamination
Before diagnostics, save Python/NumPy/Torch CPU/Torch CUDA RNG states; after all 8 anchor renders, restore them byte-for-byte. Assert a hash of all RNG states before/after restore matches. Do not sample random values inside diagnostics.

### Identity assertions before first probe
- trainer source `/root/epic05_p12/publication_trainer_p12.py` SHA256 = `675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5`
- C0 extension SHA256 = `d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11`
- C0 core SHA256 = `935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e`
- fused SSIM binary SHA256 = `dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb`
- packed=false, tile=16, eps2d=0.3, classic, 1080p, seed=42.

## Pilot first
Do `room` only and diagnostic iterations 15000,15250,15500 to validate snapshot structure. This pilot is instrumentation validation only, not scientific evidence. Confirm:
1. each anchor emits 3 NPZ files;
2. `flatten_ids` range is `[0,N)`;
3. `isect_offsets` are monotone and last tile end equals `len(flatten_ids)`;
4. running analysis produces lag-250 and lag-500 rows;
5. training trajectory without diagnostics is unchanged at a short paired replay checkpoint.

Then run the preregistered full X1/X2 window.

## Analysis
```bash
python /root/epic05_p13x/p13x_analyze_x12.py \
  --snapshots /root/epic05_p13x/artifacts/x12_snapshots \
  --scene room \
  --out /root/epic05_p13x/artifacts/P13X_X12_room.json
```

Repeat for bicycle and garden. Aggregate only after all three are complete.
