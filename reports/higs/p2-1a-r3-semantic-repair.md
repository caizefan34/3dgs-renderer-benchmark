# P2-1A-R3 — Final Semantic Correctness Repair

**Verdict:** `P2_1A_NATIVE_HIERARCHY_DROP`

R3 kept the prescribed F9 → macro partition → segmented macro sort → 32-G masks → active queue → native raster → post-compose path. No benchmark was run.

## Forensic result

The first divergent quantity is the actual native per-splat weight/effective queue support. On room tile 0, pixel [15, 15], the first traced gid 870795 has BASE weight `0.219429872` and R3 one-hot native weight `0`. The first 20 traces contain 18 mismatches. q and reconstructed alpha agree before the queue.

F9 conics are inverse-conic coefficients: `q = a*dx² + 2*b*dx*dy + c*dy²`. R3's Cholesky form reconstructs this within the FP32 reassociation envelope for all radii-valid samples. The prior non-SPD classification included uninitialized values of already culled rows.

R3 aligned the alpha cap and terminal contributor rule, but full-frame RGB remains off: room `0.977943`, bicycle `0.323744`, garden `0.185645`.

No further rescue is authorized. Timing is not authorized.
