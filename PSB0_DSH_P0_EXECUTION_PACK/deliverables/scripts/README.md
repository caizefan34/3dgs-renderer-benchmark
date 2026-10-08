# PSB0-P0 scripts (run on EPIC-05, CPU-only, no CUDA)

psb0_reference_oracle.py  — Gate 1 + Gate 5: associativity + scan-vs-replay gradient parity (float32)
psb0_operation_model.py   — Gate 4: state-recovery FLOP/memory/depth model vs native stride-32 replay, real per-scene visit counts

Run:
  scp scripts/*.py EPIC-05:/tmp/
  ssh EPIC-05 "python3 /tmp/psb0_reference_oracle.py | tee /tmp/psb0_oracle_out.json"
  ssh EPIC-05 "python3 /tmp/psb0_operation_model.py  | tee /tmp/psb0_opmodel_out.json"

Dependencies: numpy (CPU only). No torch, no CUDA, no GPU.
