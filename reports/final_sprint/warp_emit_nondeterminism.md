# Warp emit backward nondeterminism control

Status: **PASS**.

Three repetitions were run for each of room, bicycle, and garden with the same
loss used by the parity test. All repetitions had identical SHA-256 hashes for
tile counts, pre-sort IDs, sorted IDs, flatten IDs, and offsets.

| Control | Largest quaternion max-abs | Largest quaternion mean-relative error |
| --- | ---: | ---: |
| baseline ↔ baseline (same prebuilt binary) | `1.102e-3` | `1.040e-3` |
| candidate ↔ candidate (same WARP_ALL binary) | `1.148e-3` | `1.071e-3` |
| candidate ↔ baseline | `4.449e-3` | `4.114e-3` |
| unmodified JIT baseline ↔ prebuilt baseline | `4.206e-3` | `3.865e-3` |

The cross-build baseline control is essential: it changes no renderer source
yet reproduces the larger cross-binary atomic-reduction variation. Candidate
versus baseline is only 1.06× that matched control and has the same tiny
mean-error scale. With exact forward structures and bitwise render identity,
the candidate deviation lies within the normal atomic/build nondeterminism
envelope.

Raw data: `experiments/final_sprint/warp_emit/raw/nondeterminism/`.
