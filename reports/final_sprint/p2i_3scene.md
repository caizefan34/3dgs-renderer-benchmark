# P2I three-scene result

Status: `NOT_RUN — DROP_BEFORE_IMPLEMENTATION`.

| Scene | Removed count-pass upper bound | Forward-gain upper bound |
| --- | ---: | ---: |
| room | 0.0297 ms | 0.201% |
| bicycle | 0.0266 ms | 0.245% |
| garden | 0.0225 ms | 0.432% |

Fused projection must add work and a write, so actual savings can only be lower.
