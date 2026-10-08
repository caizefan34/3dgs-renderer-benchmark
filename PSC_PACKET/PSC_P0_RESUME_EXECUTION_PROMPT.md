# PSC-P0 — Resume Existing Frozen Preregistration

## Mission

Resume the already-preregistered PSC-P0 experiment. Do NOT redesign the protocol, change thresholds, tune cameras, or replace the analyzer.

The scientific question is whether Projection-Space Consensus GS has a real local optimization opportunity before any production CUDA is written.

The existing packet is the authority:
- `C:\Users\36570\3dgs-renderer-benchmark\PSC_P0_Packet\`
- `C:\Users\36570\3dgs-renderer-benchmark\PSC_P0_RUN\`

## Phase R0 — Recover frozen access

From Windows PowerShell, test WSL root entry directly:

```powershell
wsl.exe -d Ubuntu-24.04 -u root -- bash -lc '
set -e
id
whoami
test -r /root/epic05_frozen/higs_c0_final30k_worktree && echo WORKTREE_READABLE
test -r /root/epic05_p12/publication_trainer_p12.py && echo TRAINER_READABLE
ls -ld /root/epic05_frozen/higs_c0_final30k_worktree
ls -l /root/epic05_p12/publication_trainer_p12.py
'
```

Do NOT:
- use sudo as `giannis`;
- request a root password before this probe;
- chmod/chown the frozen tree;
- copy over frozen artifacts;
- rebuild any extension;
- replace any `.so`.

If `wsl.exe -u root` fails, save exact stdout/stderr and retain ACCESS_FAIL.

## Phase R1 — Frozen identity verification

Use the COMPLETE expected hashes from the existing PSC-P0 preregistration packet.

Known frozen identities that must agree with the packet include:

- worktree:
  `/root/epic05_frozen/higs_c0_final30k_worktree`
- trainer:
  `/root/epic05_p12/publication_trainer_p12.py`
- room@25K checkpoint expected SHA256:
  `b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d`
- C0 extension expected SHA256:
  `d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11`
- core `gsplat_cuda.so` expected SHA256:
  `935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e`

The packet, not this handoff note, remains authoritative if it contains additional identities.

For every required artifact record:
- label
- absolute path
- expected SHA256
- observed SHA256
- file size
- PASS/FAIL

Also record:
- hostname
- `uname -a`
- Python version
- PyTorch version
- `torch.version.cuda`
- CUDA driver/runtime
- GPU name
- compute capability
- git HEAD/status for the frozen worktree

Write a fresh machine-readable identity re-probe under `PSC_P0_RUN/`.

If ANY mandatory identity fails:
- verdict/state = `PSC_P0_IDENTITY_FAIL`
- stop before measurements.

Do not classify an access failure as a hash mismatch.

## Phase R2 — Execute the existing PSC-P0 packet unchanged

Only after full identity PASS:

1. Load the exact room@25K checkpoint.
2. Use the exact preregistered 8 frozen cameras.
3. Run the preregistered timing probe.
4. Run the preregistered projection-state / integrability probe.
5. Run the preregistered leave-one-out test.
6. Preserve all raw machine-readable outputs and logs.
7. Run the existing `psc_p0_analyze.py` on the real outputs.
8. The analyzer owns the final PSC-P0 verdict.

Strictly forbidden:
- changing the 8 cameras after seeing results;
- changing thresholds;
- changing checkpoint or renderer;
- changing resolution or support semantics;
- tuning a regularizer based on observed P0 outcomes;
- replacing missing values with estimates;
- manually upgrading a FAIL to INVESTIGATE/PASS;
- claiming a wall-clock speedup from a cost proxy.

## Scientific interpretation discipline

PSC is an approximate / altered optimization mechanism unless and until equivalence is proven. Do not label it exact.

The P0 purpose is cheap falsification:
- Is useful per-view projected-state motion compatible enough across views to be represented by periodic 3D synchronization?
- Does leave-one-out behavior support cross-view generalization rather than camera-local overfit?
- Is the avoided 3D projection/VJP work large enough to justify a later implementation?

Do not write production CUDA in P0.

## Return bundle

Return, at minimum:

- exact executed packet/script hashes;
- frozen identity report;
- environment report;
- raw timing outputs;
- raw integrability outputs;
- raw leave-one-out outputs;
- analyzer stdout/stderr;
- analyzer verdict JSON;
- contamination/failure report;
- immutable manifest of returned files.

If execution is blocked after identity PASS, classify the execution failure separately from `IDENTITY_FAIL`.
