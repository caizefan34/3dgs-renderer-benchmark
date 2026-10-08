# VJS0 — Visibility/Jacobian Superstep Feasibility Oracle — REPORT

**Verdict:** `VJS0_LINEARIZATION_IMPLEMENTATION_FAIL`

## 1. Authority
- Protocol: `VJS0_PREREG.md` (FROZEN), `DSH_HANDOFF.md` (Execute VJS0).
- Nature: feasibility UPPER-BOUND probe, NOT a 3DGS-LM / 3DGS^2-TR novelty claim.
- Host: dsw-933622-64cc845c7d-lqswc (NVIDIA A100-SXM4-80GB), device cuda:0.
- Runtime: python 3.10.21, torch 2.9.1+cu128, CUDA available=True.

## 2. Identity gate (sec.1) — PASS
- C0 extension .so match_prereg_pin=True.
- core gsplat_cuda.so match_prereg_pin=True.
- checkpoints all SHA256-match: True.
  - room@10000: `9aa00f0ddf28b6c0...` match=True
  - room@15000: `b276b338f5a444d8...` match=True
  - bicycle@10000: `26bf0b959f10f0ce...` match=True
  - bicycle@15000: `273971d3867c1d2a...` match=True

## 3. Camera semantics (sec.1) — deterministic
- Rule: TRUE HOLDOUT = canonical index `i % 8 == 0`; TRAIN POOL = rest; disjoint.
  - room@10000: n=311 train=272 holdout=39 src=[20, 59, 98, 137, 175, 214, 253, 292] hold_panel=[16, 56, 96, 136, 168, 208, 248, 288]
  - room@15000: n=311 train=272 holdout=39 src=[20, 59, 98, 137, 175, 214, 253, 292] hold_panel=[16, 56, 96, 136, 168, 208, 248, 288]
  - bicycle@10000: n=194 train=169 holdout=25 src=[12, 36, 60, 84, 109, 133, 157, 181] hold_panel=[8, 32, 56, 80, 112, 136, 160, 184]
  - bicycle@15000: n=194 train=169 holdout=25 src=[12, 36, 60, 84, 109, 133, 157, 181] hold_panel=[8, 32, 56, 80, 112, 136, 160, 184]

## 4. Linearization parity gate (sec.2) — FAIL
- Requirement: build `I(theta0+delta) ~= I0 + J@delta` with `J@delta` matching FD exact-render
  (median rel err <= 1e-3, p95 <= 5e-3) over 32 frozen directions, ||delta|| = 0.25x native Adam step.
- Allowed operators: exact autograd JVP, exact autograd VJP, custom cached-Jacobian.

| cell | JVP | VJP | FD ref norm | FD inner-cost frac | FD speedup K=4 |
|---|---|---|---|---|---|
| room@10000 | BLOCKED | BLOCKED | 4.4178e-01 | 0.556 (gate<=0.15) | 1.80x (gate>=3.0) |
| room@15000 | BLOCKED | BLOCKED | 1.1553e+00 | 0.624 (gate<=0.15) | 1.60x (gate>=3.0) |
| bicycle@10000 | BLOCKED | BLOCKED | 5.0062e+00 | 0.751 (gate<=0.15) | 1.33x (gate>=3.0) |
| bicycle@15000 | BLOCKED | BLOCKED | 5.3578e+00 | 0.786 (gate<=0.15) | 1.27x (gate>=3.0) |

### Root cause
- JVP (forward-mode AD): RuntimeError: must override the setup_context staticmethod.
  The frozen C0 autograd.Function (rasterize_gaussian_higs_dynamic) ships backward only;
  it has no jvp, so torch.func.jvp / forward_ad.dual_level cannot transform it.
- VJP (functorch): same setup_context failure under torch.func.vjp.
- Custom cached-Jacobian: infeasible — J is ~6.2M (HxWx3) x ~42M (N x param) sparse;
  not materializable; per-pixel VJP needs ~6.2M backwards; reproducing C0 forward numerics
  to 1e-3 without forward-mode is not achievable. Double-backward works but yields 2nd-order
  (Hessian-vector) info, not the first-order J@delta the operator requires.
- FD reference (the parity REFERENCE, not an allowed operator) is exact but costs ~2 renders;
  as an operator it fails the sec.6 cost (<=0.15x) and speed (>=3.0x) gates.

## 5. Verdict (sec.6) — frozen analyzer owns it
- linearization_parity_pass = False.
- Per sec.6, parity fail => VJS0_LINEARIZATION_IMPLEMENTATION_FAIL.
- Per sec.8 (forbidden rescue): no adaptive K, no K outside {2,4,8}, no line search, no
  trust-threshold tuning, no new optimizer/LR, no alternate camera subset, no post-hoc Jacobian
  rebuild, no local GPU, no full training. The K-sweep (sec.3-5) was therefore NOT run.

## 6. Interpretation
- A positive VJS0 would only say a frozen exact local renderer operator may stay useful for
  several updates; it would NOT establish novelty vs 3DGS-LM / 3DGS^2-TR.
- This negative result is a hard infrastructure blocker: the frozen C0 renderer does not
  expose forward-mode AD, so the exact cheap J@delta operator the protocol needs cannot be
  built. Reopening would require a forward-mode-capable C0 build (new jvp in the kernel),
  which is outside the frozen protocol and this execution.

## 7. Outputs
- `VJS0_ENVIRONMENT.json`  sha256=`85a5493bdc682d6f...`
- `VJS0_SOURCE_IDENTITY.json`  sha256=`ef41dbbd8c4cc65d...`
- `VJS0_CAMERA_SEMANTICS.json`  sha256=`160822fd68387524...`
- `VJS0_LINEARIZATION_PARITY.json`  sha256=`75c6a4c9f78d04af...`
- `VJS0_RAW.json`  sha256=`3b7ba65e863b5523...`
- `VJS0_SUMMARY.json`  sha256=`4346c150920c9563...`
- `VJS0_VERDICT.json`  sha256=`c9c9f38efe7757bc...`
