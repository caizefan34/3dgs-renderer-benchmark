# F9-1 ¡ª Trainable Gatherless Integration

## Result

**F9_1_WEAK.** The prototype integrates a direct-master F9 producer and gatherless VJP reads, but the diagnostic end-to-end forward gain is below 10% on room, bicycle, and garden. Exactness, densification, and the prescribed 5¡Á100 protocol remain incomplete; production integration and 5K training are not authorized.

## Audit and contract

Blend backward consumes only projected/raster state. Projection VJP now resolves `gid = visible_ids[g]` before reading master means/quaternions/scales; SH VJP does the same for master means and SH coefficients. Master-gradient writes retain the existing `visible_ids` mapping. The F9 branch saves no compact `v_means`, `v_quats`, `v_scales`, `v_opacities`, or `v_SH`.

The explicit F9 contract is FP32, pinhole, RGB, uncompressed degree-3 SH; unsupported modes fall back to the baseline path.

## Diagnostic timings

| scene | forward base ¡ú F9 (ms) | forward gain | F+B base ¡ú F9 (ms) | F+B gain | derived backward delta |
|---|---:|---:|---:|---:|---:|
| room | 2.222 ¡ú 2.031 | 8.62% | 4.393 ¡ú 4.228 | 3.75% | +0.027 ms |
| bicycle | 2.817 ¡ú 2.634 | 6.51% | 6.334 ¡ú 6.176 | 2.51% | +0.025 ms |
| garden | 2.056 ¡ú 1.863 | 9.41% | 3.716 ¡ú 3.561 | 4.16% | +0.039 ms |

## Correctness and limits

F4 intersection counts and visible counts matched in the diagnostic runs (room 953,144; bicycle 1,412,193; garden 533,928). Render RGB/alpha matched exactly in these runs; gradient relative-L2 differences were small but this is only a diagnostic comparison, not the frozen SCALAR_ADJOINT/H2-BWD-2R envelope. Flatten IDs/tile offsets, densification trace, and 5K training were intentionally not claimed as passed.

The F9-0 isolated chain discrepancy is explained by different cache/allocator state and non-additivity of medians; a complete-chain CUDA event is authoritative.

See the JSON/CSV artifacts for raw samples, resource usage, consumer audit, and traffic accounting.
