# A0-v2 Bundle

Primary current moonshot after MG0/ReBudgetGS closure and provisional Q0
negative status.

Why v2:
- distinguishes intrinsic low rank (oracle SVD) from deployable low-frequency
  structure (single frozen 2D DCT basis);
- does not confuse image dL/dRGB compression with exact compositing-backward
  sufficient-state compression;
- prevents spending time on production CUDA unless exact native coefficient
  structure is both strong and cheap enough.

Execution order:
1. exact algebra audit
2. instrumentation parity
3. bucket freeze
4. coefficient dump
5. structural analyzer
6. exact reconstructed-VJP replay
7. STOP and return evidence
