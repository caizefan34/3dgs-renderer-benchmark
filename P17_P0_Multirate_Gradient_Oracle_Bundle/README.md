# P17-P0 execution bundle

This bundle tests one mechanism only:

**Can broad, expensive Gaussians use lower-resolution gradients without changing the optimizer update materially?**

Give `DSH_TASK_P17_P0.md` to DSH.

Run in parallel with P16-RB only if a clean A100 is available; do not contaminate P16 primary timing runs.

Primary verdict uses tau=2.0 only.
Tau=1 and tau=4 are sensitivity outputs and cannot rescue a failed primary gate.

The experiment is local/frozen-state:
- no full training,
- no production CUDA,
- no speedup claim.

A GO only authorizes a later P17-P1 realizability oracle.
