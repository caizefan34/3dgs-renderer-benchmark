# AB0 REPORT — AB0_CHECKPOINT_BLOCKER (Stage -1)

- Verdict: **AB0_CHECKPOINT_BLOCKER**
- Authoritative host: EPIC-05 (A100-SXM4-80GB, 8.130.30.251:1024, root)
- Execution locus: REMOTE A100. NOT local GPU.
- Identity gate: PASS. Checkpoint gate: FAIL. Analyzer: not run. Metrics: not collected.

## 1. Identity gate (PASS)
- core gsplat_cuda.so: pinned `935ea6c0...ff37e` == found `/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so` (exact SHA256).
- C0 extension `d26cccf...513f11`: built `.so` sha256 == pinned commit exactly (`/root/epic05_build_cache/c0_ext/.../experimental_gaussian_render_inference_scene_cuda.so`). c0_ext has no `.git`; identity rests on built `.so` hash (same method as SB0).
- publication_trainer.py sha256 `1b67171d...`; cameras.json room/bicycle hashes recorded.

## 2. Checkpoint gate (FAIL → blocker)
Required cells: room@5K, room@15K, bicycle@5K, bicycle@15K (frozen C0 semantics).
- Frozen C0 publication arm saved only `ckpt_30000.pt` (trainer `--checkpoints` saves final iteration only, line 658-664). No 5K/15K optimizer states.
- Available 5K/15K checkpoints are other arms and cannot be substituted:
  - D0/ReBudgetGS TRACE (densified state, different camera order).
  - N1 c0f (seed 42, 25000 iters, only 15K, different camera order).
  - GRC3D snapshots (arm=gmain).

## 3. Action
Per prereg section 1 and DSH handoff step 2, AB0 stops at Stage -1. No timing/integration/held-out metrics collected; analyzer not run.

Decision required from research lead: authorize frozen-C0 re-run to persist 5K/15K intermediates (deviation from exact-checkpoint gate), or keep blocked.
