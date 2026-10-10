# Preflight reproduction — seeds 42 & 123

CPU-only re-run of the packaged preflight, 2026-10-10. Environment: Python 3.13.13,
NumPy 2.5.3, Windows. No GPU.

## Commands
```
python3 test_properties.py
python3 cpu_oracle.py --size 40 --num 75 --seed 42  --output <scratch>/s42
python3 cpu_oracle.py --size 40 --num 75 --seed 123 --output <scratch>/s123
```

## Results
- `test_properties.py`: **PASS: 400 random SPD/opacity/anisotropy tests, no valid pixel omitted (float64).**
- Seed 42: exact_hits=18881, stream_hits=18881, row_intervals=1263, stream_candidates=23554;
  fwd/bwd errors all 0.0; status CPU_BASIC_PARITY_PASS.
- Seed 123: exact_hits=20242, stream_hits=20242, row_intervals=1254, stream_candidates=24447;
  fwd/bwd errors all 0.0; status CPU_BASIC_PARITY_PASS.

## Comparison to packaged evidence
All numeric fields match the packaged out/s42 and out/s123 exactly. Byte-level: the
packaged JSON was written on Linux (LF); on Windows Path.write_text emits CRLF, a 44-byte
difference. After CRLF→LF normalization the reproduced files are **logically byte-identical**
to the packaged files for both seeds.

The packaged MANIFEST_SHA256.json / out/*/SHA256SUMS are **self-consistent**: the packaged
file's actual SHA256 equals the value recorded in its SHA256SUMS (e.g. out/s42 file hash
10f624a879e9bc1d725ffe862f67cdd271d640931035c478d4dbd7445cb1cf12 matches).

## Integrity note on reproduction method
The packaged run_cpu_only.sh is a bash script; the equivalent Python invocations were run
directly. No modifications were made to cpu_oracle.py, test_properties.py, or any acceptance
threshold. The `[exit code: 1]` seen during one capture was a downstream `Select-Object`
pipe truncation, not an oracle failure (confirmed by the clean full run).
