# P2I opportunity gate

Result: `DROP_BEFORE_IMPLEMENTATION`.

Protocol: official 30K room/bicycle/garden checkpoints; real camera 0; tile 16;
20 warmups; 40 reps; CUDA events and synchronization. Median is primary; raw
samples, means, and standard deviations are in `experiments/final_sprint/p2i/raw`.

| Scene | Projection | Count | Scan | Emit | Sort | Raster | Forward | Count share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| room | 0.220 | 0.0297 | 0.0604 | 10.632 | 3.107 | 0.328 | 14.780 | 0.201% |
| bicycle | 0.332 | 0.0266 | 0.0573 | 6.584 | 2.651 | 0.897 | 10.880 | 0.245% |
| garden | 0.187 | 0.0225 | 0.0553 | 1.961 | 1.648 | 1.085 | 5.220 | 0.432% |

All values are ms. Emit is `no_sort - count - cumsum`; sort is
`with_sort - no_sort`. Forward is the ordered sum of stable event-timed pipeline
components, including SH. The exact extracted count branch had zero mismatches
and zero max difference against production counts in every scene. Count removal
is therefore an upper bound on forward gain; every bound is below 3%, so the
mandatory early-stop rule applies.
