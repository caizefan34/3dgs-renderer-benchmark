# VOS-GS Prior-Art Audit — FINAL (Phase 1 deliverable)

**Idea under audit (VOS):** Consumer-driven logical-time virtualization of optimizer
states with exact replay of deferred dense-optimizer transitions. Per row + attribute
maintain `(theta, m, v, tau)` where `tau` = the logical optimizer step of the current
physical state. When the attribute is next consumed by the training graph, replay all
deferred zero-gradient Adam transitions (`s = tau+1 .. t-1`, exact per-step
hyperparameters and bias correction) in registers before the consumer reads `theta`.

Audit date: 2026-09-29 (UTC). Draft audit (subagent, direct fetches) + final arXiv
API collision sweep (main session) merged into this file. Evidence labels:
**OBSERVED** = source/URL read directly; **DERIVED** = inferred from read evidence;
**UNVERIFIED** = could not access.

**VERDICT (short form):** NO_COLLISION in public prior art for "per-row timestamp +
exact replay of skipped stock dense-Adam transitions before the next parameter
consumer". Closest public art = BOGausS (per-Gaussian step counter, but a *modified*
optimizer with no replay) and AdamW-GS (per-Gaussian step decoupling + re-designed
optimizer, no replay, no dense-equivalence claim). The exact-phrase arXiv sweep
(Section 0.1) is clean. Semantic Scholar / Google Scholar remain unswept (network
rate-limits) — residual gap, scoped in Caveat 1. The mechanism itself already exists
internally in this repo (Section 10) — VOS is its systems-engineering continuation
under a different target trainer and a different gate (Caveat 3).

---

## 0. Environment / access notes (read this first)

- The `web_search` tool failed with an invalid-API-key error on every call (3 attempts,
  different queries; reproduced independently in the main session — same auth error).
  All web evidence below was gathered by **direct HTTP fetches**
  (raw.githubusercontent.com, arxiv.org, github.com, api.openreview.net) from this
  machine.
- Search engines unreachable from this network: DuckDuckGo (html+lite), Google (both
  timed out), Bing (CN instance ignores query and returns unrelated results), Semantic
  Scholar API (persistent HTTP 429, retried over several minutes). HuggingFace papers
  search ignored the query parameter (returned a fixed trending list).
- **The arXiv Atom API over HTTPS DID work** (export.arxiv.org) — the exact-phrase and
  keyword collision sweep was executed there; see Section 0.1.

### 0.1 arXiv API collision sweep (executed 2026-09-29, main session)

Method: `https://export.arxiv.org/api/query?search_query=...&max_results=5..6`,
3 s spacing between queries, `opensearch:totalResults` + titles recorded. Raw
transcripts: `vos/scripts/vos_prior_art_sweep.ps1` / `..._sweep2.ps1` (job logs in
session). All queries searched `all:` fields (title+abstract+authors+comments).

| Query | Results | Relevant? |
|---|---:|---|
| `"deferred optimizer state"` | **0** | — |
| `"virtual optimizer state"` | **0** | — |
| `"logical optimizer state"` | **0** | — |
| `"lazy adam"` | **0** | — |
| `"lazy adam" AND "gaussian splatting"` | **0** | — |
| `"gaussian splatting" AND "bias correction" AND sparse` | **0** | — |
| adam AND `"deferred update"` | **0** | — |
| adam AND materialize AND deferred | **0** | — |
| `"per-parameter" AND timestamp AND optimizer` | **0** | — |
| `"optimizer state" AND "gaussian splatting" AND visible` | **0** | — |
| `"selective adam"` | **0** | — |
| optimizer AND replay AND skip AND adam | **0** | — |
| `"amortized optimizer"` | 5 | none relevant (amortized-optimization literature) |
| `"gaussian splatting" AND optimizer AND skip` | 4 | none implement deferral+replay (SkipGS = backward-pass gating; TurboGS = sparse pixel sampling; CAT-3DGS = compression; DreamMapping = unrelated) |
| `"gaussian splatting" AND optimizer AND invisible` | 5 | watermarking/removal/compression papers only |
| visibility AND adam AND `"gaussian splatting"` | 1 | OccFusion (unrelated) |
| `"sparse adam" AND "gaussian splatting"` | 1 | **"A Step to Decouple Optimization in 3DGS" = AdamW-GS** — resolved, see Section 11.5 |

**Sweep conclusion (OBSERVED):** zero arXiv hits for the VOS collision phrasings; the
two adjacent hits (AdamW-GS, SkipGS) are resolved in Sections 11.5/11.6 and are not
collisions. Note the arXiv API searches metadata (title/abstract), not full text — a
full-text sweep remains open (Caveat 1).

- TensorFlow source/docs: `raw.githubusercontent.com/tensorflow/tensorflow/master/.../optimizer_v2/adam.py`
  fetched successfully (20,383 bytes) but contains **no** `LazyAdam`/`SparseAdam`
  class (verified by string search). The tf docs page (tensorflow.org) timed out
  repeatedly. LazyAdam semantics are therefore **UNVERIFIED** in this audit.
- **AdamW-GS: RESOLVED** — found via the arXiv API sweep as "A Step to Decouple
  Optimization in 3DGS" (arXiv 2601.16736v5, accepted ICLR 2026). See Section 11.5.
- All local evidence below was read directly from files in this repo.

---

## 1. Dense torch.optim.Adam (the semantics VOS claims to preserve)

**Source (OBSERVED, installed torch 2.13.0+cu130):**
`C:\Users\36570\miniconda3\Lib\site-packages\torch\optim\adam.py`

- Per-parameter update applies to **every element** of every parameter with a
  non-None grad: `exp_avg.lerp_(grad, 1-beta1)` (line 456),
  `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)` (line 475),
  bias-corrected step `param.addcdiv_(exp_avg, denom, value=-step_size)` (line 546).
- A row with gradient 0 still receives: `m *= beta1`, `v *= beta2`, and
  `theta -= lr/(1-beta1^t) * m / (sqrt(v)/(1-beta2^t) + eps)` — i.e. a **nonzero
  parameter move** (unless m is already 0). The `step` counter advances every call.
- **Answers:** (a) yes, updates zero-grad rows every step; (b) this IS the dense
  semantics; (c) no deferral/replay (nothing to defer); (d) per-parameter global
  `step` counter only, no per-row timestamps.
- The fused path (`torch.optim.Adam(fused=True)`) does the same math batched;
  `benchmark\higs_masked_adam.py` lines 19-26 mirror its exact rounding.

## 2. PyTorch torch.optim.SparseAdam

**Source (OBSERVED, installed torch 2.13.0+cu130):**
`C:\Users\36570\miniconda3\Lib\site-packages\torch\optim\sparse_adam.py` (lines 132-161, docstring) and
`C:\Users\36570\miniconda3\Lib\site-packages\torch\optim\_functional.py` (lines 24-84).

- Docstring (verbatim, lines 139-143): "SparseAdam **approximates** the Adam algorithm
  by masking out the parameter and moment updates corresponding to the zero values in
  the gradients. Whereas the Adam algorithm will update the first moment, the second
  moment, and the parameters based on all values of the gradients, **SparseAdam only
  updates the moments and parameters corresponding to the non-zero values** of the
  gradients."
- Implementation (OBSERVED, `_functional.py` lines 62-84): only sparse-index
  (nonzero) entries of `exp_avg` / `exp_avg_sq` / `param` are touched. Zero entries of
  the sparse-grad layout are never updated: **no decay, no bias-corrected parameter
  move** for un-touched entries.
- Step counter `state["step"] += 1` (line 112) is **global** (per parameter), not
  per-row: bias correction uses the global count, while rows are updated only when
  they appear in the sparse grad.
- Intended for `nn.Embedding(sparse=True)`-style gradients (dense params, sparse-grad
  layout). Not a 3DGS tool; semantics identical in kind to the 3DGS sparse optimizers
  below.
- **Answers:** (a) does NOT update zero-grad rows; (b) does NOT preserve dense-Adam
  semantics (documented as an approximation); (c) no deferral, no replay; (d) no
  per-row logical time (global step counter only).

## 3. TensorFlow LazyAdam

**Status: UNVERIFIED in this audit.**

- Attempted: tf.org docs page (timeout, 3 attempts); TF GitHub master
  `tensorflow/python/keras/optimizer_v2/adam.py` (fetched, 20,383 bytes, **no**
  `LazyAdam` or `SparseAdam` class present — string search); TF v2.9.0 same path (20,343
  bytes, no match); PyPI (no sdist; wheels only). The class evidently lives in a
  different module path not identified here.
- What is known from the task brief and prior project notes: LazyAdam is the
  Keras-level sparse-Adam variant used with `nn.Embedding`-like sparse gradients;
  community reports describe it as skipping updates for un-touched rows (no decay).
  **Treat all LazyAdam specifics as UNVERIFIED until the source is read.**
- **Answers:** (a)–(d) all UNVERIFIED (best-guess: skip, no dense semantics, no
  replay, no per-row time — same family as SparseAdam).

## 4. SparseGaussianAdam — official graphdeco 3DGS ("accel" rasterizer)

**Local mirrors (OBSERVED — identical code in 3 places):**
- `public_baselines\taming-3dgs-rasterizer\submodules\diff-gaussian-rasterization\cuda_rasterizer\adam.cu`
  (lines 9-38 kernel; lines 26-37 are the whole update, gated by `if (tiles_touched[g_idx])`)
- same tree, `diff_gaussian_rasterization\__init__.py` lines 236-263 (Python wrapper)
- `public_baselines\fastgs\submodules\diff-gaussian-rasterization_fastgs\cuda_rasterizer\adam.cu`
  (byte-identical kernel) and its `__init__.py` lines 245-272
- `experiments\training_solver_design\fastgs_scalar_adjoint_trial\diff_gaussian_rasterization_fastgs\__init__.py`
  line 245 (further copy)

**Official upstream (OBSERVED via raw.githubusercontent.com, graphdeco-inria/gaussian-splatting @ main):**
- `scene/gaussian_model.py`: `try: from diff_gaussian_rasterization import SparseGaussianAdam`
  (line 26); used as `self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)` when
  `optimizer_type == "sparse_adam"` (local mirror `reference\graphdeco\gaussian_model.py`
  lines 192-199, with fallback to dense Adam if the accel rasterizer is absent).
- `train.py` (official, fetched): `visible = radii > 0; gaussians.optimizer.step(visible, radii.shape[0])`
  (local mirror `reference\graphdeco\train.py` lines 180-182).
- The upstream `submodules/diff-gaussian-rasterization/...` submodule path 404'd on
  main (submodule not checked out in that path); the kernel above is the published
  "accel" variant, and the local taming mirror is the SIGGRAPH-Asia-2024 Taming 3DGS
  release of the same code.

**Mechanism (OBSERVED):**
- Kernel launches one thread per element; per Gaussian row `g_idx = p_idx / M`;
  the entire update — `m = b1*m + (1-b1)*g`, `v = b2*v + (1-b2)*g*g`,
  `theta -= lr*m/(sqrt(v)+eps)` — executes **only if `tiles_touched[g_idx]` is true**.
- For invisible rows: **no moment decay, no parameter move, no bias correction**
  (there is no bias correction at all — the kernel omits it entirely).
- Note: the kernel has **no bias correction** even for visible rows (compare with
  dense Adam's `1/(1-beta1^t)` factors). This is an additional, separate
  non-equivalence vs dense Adam, independent of the masking.
- Visibility mask = `radii > 0` (projected screen radius), per `train.py`.
- The Taming 3DGS README documents this explicitly: "Sparse adam is an alternate
  implementation for the Adam optimizer that applies gradients only to the gaussians
  that are visible. **Note that this can change the training behaviour**, whereas all
  other changes don't affect the numerical results."
  (`public_baselines\taming-3dgs-rasterizer\README.md` lines 40-41, OBSERVED)

**Answers:** (a) does NOT update invisible rows (skips entirely, no decay);
(b) does NOT preserve dense-Adam semantics (author-acknowledged; also no bias
correction); (c) no deferral, no replay; (d) no per-row logical time — the Python
wrapper's `state['step']` (line 254) is a global counter that is **never used by the
kernel**.

## 5. gsplat SelectiveAdam

**Source (OBSERVED):**
- `public_baselines\gsplat\gsplat\optimizers\selective_adam.py` lines 21-101
  (Python wrapper; docstring lines 22-35: "This is one of the two optimizers
  mentioned in the Taming3DGS paper")
- `public_baselines\gsplat\gsplat\cuda\csrc\AdamCUDA.cu` lines 35-71 (kernel)

**Mechanism (OBSERVED):**
- Kernel (lines 49-59): `if (valid != nullptr && !valid[g_idx]) return;` — the whole
  per-element update is skipped when the row's `valid` bit is false.
- Same update math as SparseGaussianAdam (`m = b1*m+(1-b1)*g`,
  `v = b2*v+(1-b2)*g*g`, `theta -= lr*m/(sqrt(v)+eps)`), **no bias correction**,
  no per-row step.
- The `adam` CUDA op is also exposed directly
  (`public_baselines\gsplat\gsplat\cuda\_wrapper.py` lines 419-432) and used with
  `sparse_grad`-style masks; gsplat docs reference `torch.optim.SparseAdam` as the
  canonical sparse optimizer (`public_baselines\gsplat\gsplat\rendering.py` line 361).
- The local baseline freeze record confirms usage:
  `baseline_closure\BASELINE_FREEZE.json` line 129: "Adam (or SparseAdam/SelectiveAdam),
  lr * sqrt(batch_size*world_size)..." and line 163 (graphdeco): "**Sparse Adam is NOT
  numerically equivalent to dense Adam**: invisible Gaussians skip
  exp_avg/exp_avg_sq decay and no bias correction is applied (README: 'this can
  change the training behaviour'). P8 matched arm MUST use the dense default optimizer."

**Answers:** (a) does NOT update masked-out rows; (b) no dense-Adam equivalence
(no decay, no bias correction); (c) no deferral/replay; (d) no per-row logical time.

## 6. Faster-GS (nericg-project/faster-gaussian-splatting, arXiv 2602.09999)

**Sources (OBSERVED):**
- Paper: https://arxiv.org/abs/2602.09999 "Faster-GS: Analyzing and Improving Gaussian
  Splatting Optimization" — abstract fetched directly.
- Full text: https://arxiv.org/html/2602.09999v1 — fetched directly (100,239 chars
  cleaned text). Key passages quoted in Section 6.2 below.
- Local mirror: `public_baselines\faster-gs` (commit 117bdbacfd7df6c1f06d9d959b1f5ab02d4d3847
  per `experiments\training_solver_design\VISIBILITY_LAZY_ADAM_OPPORTUNITY.md` line 231-235).

**6.1 FusedAdam (main branch, OBSERVED locally):**
- `public_baselines\faster-gs\FasterGSCudaBackend\FasterGSCudaBackend\torch_bindings\adam.py`
  lines 6-36: `class FusedAdam(torch.optim.Adam)` — a **dense** optimizer: iterates
  all params, `state['step'] += 1`, calls `_C.adam_step(grad, param, exp_avg,
  exp_avg_sq, step, lr, b1, b2, eps)`.
- `...\adam\src\adam.cu` lines 10-34: kernel covers **all** `n_elements`
  (`if (idx >= n_elements) return;` only); update math includes bias correction
  (`bias_correction1_rcp`, `bias_correction2_sqrt_rcp` computed on host, lines 54-56)
  and explicitly models itself on PyTorch's fused Adam
  (comment line 9: "based on .../torch/csrc/api/src/optim/adam.cpp#L72-L142").
- Paper Sec. 3.3 (OBSERVED, HTML text): "...we develop our own Adam implementation in
  which we **precisely match the behavior of the PyTorch implementation** while
  optimizing away all unnecessary overhead."
- So Faster-GS's *main* optimizer is dense-Adam-exact (modulo floating-point
  reassociation). It does **not** skip invisible rows by default.

**6.2 "Skip invisible" variant (paper Sec. 3.3 + Table 3, OBSERVED):**
- Paper text (verbatim, cleaned): "A recent idea by Mallick et al. [56] is to skip
  updates for these invisible Gaussians, which fits our fused design exceptionally
  well. We see this as an optional extension for further acceleration, however, as
  **it can cause inconsistencies and performance regressions** compared to the
  original 3DGS implementation."
- Paper Table 3 (Mip-NeRF 360, 5 outdoor scenes): Full 24.72 dB / 5m31s;
  "+ fused updates" 24.73 dB / 5m04s; **"+ fused updates (skip invisible)" 24.59 dB
  / 3m03s** (−0.14 dB, −2m28s); "+ fused updates (SH degree=0)" 24.38 dB / 2m24s.
- Local prior-art note (OBSERVED): `experiments\training_solver_design\VISIBILITY_LAZY_ADAM_OPPORTUNITY.md`
  lines 217-235 reproduces exactly this table and adds: "Exact deferred updates would
  seek that saved memory traffic **without accepting the changed Adam trajectory**, but
  neither similar speed nor recovered quality has been demonstrated here."
- The `FasterGSFused` branch (backward/optimizer fusion into the rasterizer kernel) is
  referenced in the local README notes (VISIBILITY_LAZY_ADAM_OPPORTUNITY.md lines
  232-235) but has **not** been source-audited in this workspace (OBSERVED note; the
  branch itself UNVERIFIED).

**Answers:** (a) main FusedAdam updates ALL rows (dense, exact); the optional
"skip invisible" variant skips invisible rows with no decay; (b) main variant
preserves dense-Adam semantics by construction (paper claim + kernel inspection);
skip variant does not; (c) **no deferral/replay in either variant** — skipped rows
are simply left stale forever (their m/v/theta are frozen, never caught up);
(d) no per-row logical time — global `state['step']` only.

## 7. FastGS (public_baselines\fastgs)

**Source (OBSERVED):**
- `public_baselines\fastgs\scene\gaussian_model.py` lines 198-244.
- `public_baselines\fastgs\train.py` lines 160-167.

**Mechanism (OBSERVED):**
- Two optimizer types. `"default"`: dense `torch.optim.Adam` for xyz/f_dc/opacity/
  scaling/rotation **plus a second dense Adam** (`shoptimizer`) for `f_rest`, stepped
  on a schedule: every step <15000; every 32nd step 15000–20000; every 64th step
  >20000 (`optimizer_step`, lines 225-244). This is **attribute- and schedule-decoupled
  dense Adam** — different parameters get different update frequencies, but every
  update that does happen is full dense Adam (with bias correction, inherited from
  torch.optim.Adam). No rows are ever skipped; the "sparsity" is temporal
  (infrequent SH updates), not per-row.
- `"sparse_adam"`: the SparseGaussianAdam from Section 4 (line 211:
  `self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)`), driven by
  `visible = radii > 0` (train.py line 165). Same skip-semantics as Section 4.
- Note the design: in `sparse_adam` mode FastGS merges the two groups into one
  SparseGaussianAdam; in `default` mode SH gets its own optimizer on a slower clock.
  Neither mode defers transitions or tracks per-row logical time.

**Answers:** (a) default: updates all rows (dense); sparse_adam: skips invisible
rows (no decay); (b) default preserves dense semantics (per-group); sparse_adam
does not; (c) no deferral/replay anywhere; (d) no per-row logical time (global step
counters in torch state).

## 8. Speedy-Splat (public_baselines\speedy-splat)

**Source (OBSERVED):**
- `public_baselines\speedy-splat\train.py` lines 195-196:
  `gaussians.optimizer.step(); gaussians.optimizer.zero_grad(set_to_none=True)`.
- grep for `optimizer|Adam|sparse` in that train.py: only the two lines above.
- The local speedy-splat checkout is the **training-speed** repo (fused SH, fused
  backward, etc.); its optimizer path is stock dense Adam via the graphdeco
  `GaussianModel.training_setup` (no `sparse_adam` branch in this train.py).
- The task brief asked for "SparseGaussianAdam (official graphdeco optimizer)" in
  speedy-splat — that attribution is imprecise: the official SparseGaussianAdam lives
  in the graphdeco *accel rasterizer* (pip `diff-gaussian-rasterization[3dgs_accel]`),
  which is what the Taming 3DGS repo and FastGS embed as a submodule. Speedy-splat's
  local train.py does not reference it. (OBSERVED: no `SparseGaussianAdam` symbol in
  `public_baselines\speedy-splat\`.)

**Answers:** (a) dense, updates all rows; (b) preserves dense-Adam semantics;
(c) none; (d) none.

## 9. Local repo: HiGS masked Adam (benchmark\higs_masked_adam.py)

**Source (OBSERVED, full file read, 196 lines):**
- `benchmark\higs_masked_adam.py` — "Fused cull-masked Adam step for HiGS training"
  (Round-60 lever). Docstring lines 1-31 state the design precisely:
  - Runs **the exact torch fused-Adam math** (`m = b1*m+(1-b1)*g`,
    `v = b2*v+(1-b2)*g^2`, `p -= lr/bc1 * m / (sqrt(v)/sqrt(bc2)+eps)`)
    **only on rows where a per-Gaussian mask is True** (kernel line 56:
    `if (!mask[row]) return;`).
  - "culled Gaussians are **frozen completely (params and moments unchanged)**,
    which **changes dynamics vs stock Adam** only for out-of-view rows (stock Adam
    keeps decaying their moments with zero grads)." (lines 13-17)
  - Bias corrections computed in double and cast to float to mirror torch's fused
    kernel; measured equivalence: "params within 1-2 float32 ulps of torch's fused
    step (max rel diff ~2e-7), state within ~1 ulp on a small fraction of elements,
    **frozen rows bit-identical**." (lines 22-26)
  - `masked_adam_step(opt, mask)` (lines 152-196): lazily creates
    `state["step"] = 0`, `exp_avg`, `exp_avg_sq`; increments a **global**
    `state["step"]` per call; computes `bc1 = 1 - beta1^step`,
    `bc2_sqrt = sqrt(1 - beta2^step)`, `step_size = lr/bc1` on host, then launches
    the masked kernel.

**Key semantic points (OBSERVED + DERIVED):**
- For masked-out rows: **frozen** — no decay, no move. Same skip-family as
  SparseGaussianAdam/SelectiveAdam, but with **bias correction included** for
  updated rows (unlike SparseGaussianAdam/SelectiveAdam which omit it), and the
  per-row update is bit-close to torch fused Adam.
- There is a subtlety the file itself does not discuss (DERIVED): the **global**
  `state["step"]` is used for bias correction of rows that update *rarely*. A row
  first updated at logical step t gets `bc1 = 1 - beta1^t` as if it had been updated
  at every step — which coincidentally matches dense Adam's bias correction for that
  row (dense Adam also uses the global step count for every row). So on the rows that
  DO update, the bias correction is dense-Adam-consistent; on the rows that don't,
  everything is frozen. **No replay of the frozen interval exists.**
- Probe scripts: `scripts\higs\probe_masked_adam.py`,
  `scripts\higs\probe_masked_adam_speed.py`,
  `scripts\higs\run_round60_masked_adam.sh` (OBSERVED via glob; not read in depth).
- Related local evidence: `artifacts\higs-e2-p0\adam_traffic.json`,
  `reports\higs\e2-p0-vjp-adam-opportunity.md` (OBSERVED via glob).

**Answers:** (a) does NOT update masked-out rows (frozen, no decay); (b) does NOT
preserve dense-Adam semantics for those rows (explicitly "changes dynamics"); for
updated rows it is bit-close to dense fused Adam; (c) no deferral/replay; (d) no
per-row logical time (global `state["step"]`).

## 10. Local repo: the exact-catch-up oracle (experiments\training_solver_design)

This is the single most important local finding for the collision check: **the VOS
mechanism — per-row timestamp + exact replay of skipped zero-gradient Adam steps —
was already prototyped and numerically validated in this very repo** (internal work,
pre-dating this audit; it is the direct ancestor of the VOS idea).

**Sources (OBSERVED, full files read):**
- `experiments\training_solver_design\lazy_adam_oracle.py` (119 lines) — CPU FP64
  reference:
  - `scalar_adam_step(theta, m, v, gradient, step, lr, beta1, beta2, eps)` (lines
    14-20): exact dense Adam step **with bias correction** (`m/(1-beta1^step)`,
    `sqrt(v/(1-beta2^step))`).
  - Main loop (lines 50-69): for each coordinate, tracks `last_step[i]`; when the
    gradient at `step` is nonzero, it first **replays all skipped zero-gradient
    steps** `for skipped in range(last_step[i]+1, step)` with the exact per-step
    `learning_rates[skipped]` (a decaying schedule), then applies the real step and
    sets `last_step[i] = step`. This is **exactly the VOS replay rule** (deferred
    zero-gradient transitions replayed before the next consumer).
  - Final flush (lines 71-76): replays the tail for all coordinates.
  - Compared against a real `torch.optim.Adam` (foreach=False, fused=False) over
    500 steps × 64 coordinates with ~25% nonzero-gradient coordinates.
  - `zero_gradient_displacement_bound` (lines 23-29): a closed-form bound on how far
    a frozen row can drift over `k` zero-grad steps — used as a conservative
    "is it safe to keep this row unmaterialized?" gate. Validated on 10,000 random
    cases (`assert abs(theta) <= bound*(1+1e-12) + 1e-15`, line 102).
- `experiments\training_solver_design\lazy_adam_oracle.json` (OBSERVED):
  `"max_absolute_errors": {"first_moment": 1.39e-17, "second_moment": 0.0,
  "theta": 4.44e-16}`, `"observed_fraction": 0.2498`, `"zero_gradient_bound_trials":
  10000`, `"max_actual_to_bound_ratio": 1.0000000000000002`.
  → The replay scheme is **numerically exact** (to FP64 roundoff) vs dense PyTorch
  Adam, including with a **time-varying learning rate** (the schedule
  `0.02*(0.04^(t/500))` is passed per skipped step — i.e. per-step hyperparameters
  are honored in the replay, exactly as VOS specifies).
- `experiments\training_solver_design\VISIBILITY_LAZY_ADAM_OPPORTUNITY.md`
  (OBSERVED, 251 lines) — the design doc for the GPU version:
  - "maintain a logical optimizer-step timestamp `s` and physically materialize
    `(theta,m,v)` only before a camera might need the row, before any
    topology/opacity/score event that reads its current parameters, and at export"
    (lines 55-58) — this is the VOS "consumer-driven" materialization, word for word.
  - Gives the exact deferred transition (lines 62-66):
    `m_j = beta1*m_(j-1); v_j = beta2*v_(j-1); theta_j = theta_(j-1) -
    eta_j*(m_j/(1-beta1^j))/(sqrt(v_j/(1-beta2^j)) + eps)`.
  - Handles FastGS's late-stage gradient accumulation (32/64-view) by noting "the
    debt clock advances only on actual Adam steps" (lines 68-70).
  - Correctness gates: a world-space 3-sigma sphere gate FAILED (401 false
    negatives at factor 1, missed cases at factors 1/1.5/2) and was replaced by a
    renderer-derived screen-envelope gate with zero false negatives on 272 cameras
    (lines 89-101, 122-141).
  - Opportunity analysis: Adam is 22.27% of 44-step ordinary-step CUDA time; even
    erasing ALL Adam work caps ordinary-step speedup at 1.286×; the 35K-vs-30K wall
    gap is 29.06 s while the optimistic Adam erasure saves 12.7–18.6 s (lines
    14-21, 105-120). Verdict: "This candidate has **not** earned an isolated GPU
    implementation" (line 117).
  - Its own prior-art boundary (lines 217-242) cites Faster-GS §3.3 + Table 3,
    Taming 3DGS, BOGausS ("per-Gaussian optimizer age" — EUSIPCO 2025, DOI
    10.23919/EUSIPCO63237.2025.11226626), and ACE-GS; and concludes: "A possible new
    contribution is **conservative visibility plus exact deferred dense-Adam
    settlement** in a topology-changing 3DGS trainer, **if a complete source audit
    confirms priority** and the GPU/full-training gates pass. Until then this remains
    a hypothesis."
- `experiments\training_solver_design\NEXT.md` lines 713-758 (OBSERVED): same stage,
  current status: "The CPU FP64 zero-gradient Adam catch-up matches PyTorch Adam to
  4.44e-16 max parameter error, and its scalar displacement bound passed 10,000
  random cases... Exact deferred settlement and conservative visibility remain
  untested on GPU and their **originality is unconfirmed**."
- `experiments\training_solver_design\newborn_adam_audit.py` +
  `newborn_adam_evidence.json` (OBSERVED): measures how the global-step bias
  correction affects *newborn* (densification) rows — fresh rows with zero moments
  at step ~1985 take ~45× smaller updates than same-age rows; relevant to VOS
  because densification/cloning/splitting are topology events that VOS must flush
  exactly (VISIBILITY_LAZY_ADAM_OPPORTUNITY.md line 72-75).
- SADGS hybrid note (OBSERVED):
  `experiments\training_solver_design\SAD_HYBRID_MASK_SOURCE_AUDIT_20260927.md`
  line 12 documents a third-party (LinjieLyu/SADGS) `SparseGaussianAdam` wrapper
  (submodules/diff-gaussian-rasterization_structgs `__init__.py` lines 254-276) that
  "forwards the supplied visibility mask to `_C.adamUpdate`" — i.e. the same
  skip-semantics family (the upstream file itself not fetched; the wrapper
  description is from the local audit note, OBSERVED note / UNVERIFIED upstream).

**Answers:** (a) the oracle does NOT let frozen rows decay — it replays them
exactly before the next update (the VOS rule); (b) YES, preserves dense-Adam
semantics to 4.44e-16 in the oracle (CPU FP64, scalar, time-varying lr); (c) YES,
defers AND replays (this is the whole point of the oracle); (d) YES, per-row
`last_step[i]` timestamp. **Scope limits (OBSERVED):** CPU-only, scalar (not
batched), no GPU kernel, no topology (clone/split/prune) handling in the oracle,
no full-training run; the GPU conservative-visibility gate is still an open problem
(sphere gate failed; screen-envelope gate passed one 272-camera sample only).

## 11. BOGausS: Better Optimized Gaussian Splatting (EUSIPCO 2025) — **CLOSEST PUBLIC PRIOR ART**

**Source (OBSERVED — full accepted manuscript in repo):**
`experiments\training_solver_design\topology_sources\bogauss.txt` (952 lines, "ACCEPTED
MANUSCRIPT / CLEAN COPY"). Authors: S. Pateux, M. Gendrin, L. Morin, T. Ladune,
X. Jiang (Orange Innovation / Univ Rennes, INSA Rennes). DOI
10.23919/EUSIPCO63237.2025.11226626 (resolves to ieeexplore.ieee.org/document/11226626
— paywalled, 0 bytes body; HAL copy cited locally: https://hal.science/hal-05444229).
The manuscript text below is quoted from the local copy.

**What it does (OBSERVED, manuscript §IV, lines 204-335):**

- §IV-A "Proposed optimizer" (lines 204-258): a **modified Adam** (Eq. 3):
  `b_g1_t = b_g1_{t-1} + α_{1,t}(∇θL_t − b_g1_{t-1})`,
  `b_g2_t = b_g2_{t-1} + α_{2,t}((∇θL_t)² − b_g2_{t-1})`,
  `θ_{t+1} = θ_t + η_θ · Δθ · b_g1_t / (b_g2_t + ε)`,
  with `α_{s,t} = max(1/t, 1−β*_s)` (lines 221-224) — a time-adaptive blend factor
  that interpolates between the previous filtered state and the fresh gradient. The
  authors call the result a "modified sparse estimator" (NEWBORN_RESULTS.md line 39,
  local note OBSERVED).
- §IV-C "Sparse and unbiased states updates" (lines 284-297, **the key passage**):
  "As in [4] we propose to use **Sparse Adam** to perform parameters update only on
  Gaussians that contribute to the rendered image being optimized... However, unlike
  biased estimator used in [4], the BoGausS proposed modified Adam's optimizer defined
  by (3) allows ensuring **unbiased states** (b_g1_t, b_g2_t) estimation... **Indeed
  instant t in (3) is Gaussian dependent and only incremented when a Gaussian is
  updated. t then naturally refers to Gaussian's lifespan. This additional state in
  optimizer is needed to ensure unbiased states.**"
  → **BOGausS maintains a per-Gaussian step counter `t` (per-Gaussian logical time)**
  used for bias correction — this is the *same structural idea* as VOS's `tau`, but
  its purpose is unbiasedness of the modified estimator, not deferred replay.
- §IV-D "States inheritance after Gaussian splitting" (lines 298-335, Eq. 4):
  children inherit `t_child = α_t × t_mother`, `g_child = α_g × g_mother`,
  `g2_child = (α_g)² × g2_mother` with `α_t = 1, α_g = 0.2`. This is **topology-aware
  optimizer-state transport** — VOS also has to handle topology events (clone/split/
  prune), so this is a direct overlap in *problem space* (BOGausS solves the
  inheritance sub-problem with a different, non-exact rule).
- §III (lines 180-203): precision model `Δ_i = α·tr(Σ^{(XYZ)}_i)^{−0.5}` used as the
  `Δθ` update scale for position/size parameters (precision-aware updates).
- Ablation (TABLE II, lines 860-885, Garden 1.01M): full BOGausS 27.81 dB / 0.861 /
  0.132; "no Sparse Adam" 27.49 dB (−0.32); "no states inheritance at split" 27.77 dB
  (−0.04); "no scaled updates" 27.51 dB (−0.30). Local note (NEWBORN_RESULTS.md
  line 37, OBSERVED): "BOGausS 已经提出按高斯 lifespan 的优化器状态更新和父子状态继承，
  Eq.(4) 明确给出一阶矩乘 α、二阶矩乘 α²，经验 α=0.2" — i.e. the repo's own earlier
  audit already flagged this.

**Answers:** (a) does NOT update un-touched rows (Sparse-Adam skip, as in Taming
[4] of their bibliography); (b) does NOT preserve stock dense-Adam semantics — it is
a *different* optimizer (modified Adam with `α_{s,t} = max(1/t, 1−β*)` blend and
precision-scaled steps); its "unbiased" claim is about its own modified recurrence,
not about reproducing dense Adam; (c) **NO deferral, NO replay** — un-touched rows
are simply skipped (no catch-up of the skipped interval; grep for
replay/catch-up/deferred/materializ/skip in the full manuscript returns zero
matches); (d) **YES — per-Gaussian step counter `t`** ("Gaussian's lifespan"),
incremented only when the Gaussian is updated, used for bias correction; plus
parent→child inheritance of that counter at splits.

**VOS vs BOGausS delta (the most important delta in this audit):**

| Aspect | BOGausS | VOS |
|---|---|---|
| Per-Gaussian logical time | YES (`t` = lifespan) | YES (`tau`) |
| Purpose of the counter | Unbiased bias correction for its *modified* Adam | Exact replay of *stock* dense-Adam transitions |
| Un-touched rows | skipped, no decay, no move, no catch-up | deferred, **replayed exactly** before next consumer |
| Optimizer math | modified Adam (`α_{s,t} = max(1/t, 1−β*)`, precision-scaled) — deliberately different from dense Adam | stock dense Adam, bit-exact (4.44e-16 in oracle) |
| Topology (splits) | explicit inheritance rule (α_t=1, α_g=0.2) | requires flush/ordering rules (open; newborn_adam_audit.py) |
| Claim | "unbiased states" for its own estimator | "dense-Adam-equivalent" trajectory |

So: BOGausS **anticipates the per-Gaussian-timestamp component** of VOS (and does so
for a related goal — unbiased sparse updates — in 2025), but it does **not** implement
deferred-transition replay, does not preserve stock dense-Adam semantics (it changes
the optimizer), and its whole thesis is that the *modified* estimator is what you
want. VOS's exactness claim (replay the exact dense-Adam recurrence) is therefore
still a real delta — but the "per-row logical time" idea is **not** novel against
BOGausS, and the local repo's own earlier audits already knew this
(SIBLING_COORDINATES_DESIGN.md line 14: "BOGausS 已有 lifespan 和父子状态继承，因此
它们不能作为主创新"; NEXT.md lines 1052-1054: "BOGausS (EUSIPCO 2025) directly
covers per-Gaussian lifespan and parent-to-child [inheritance]").

### 11.5 AdamW-GS — "A Step to Decouple Optimization in 3DGS" (arXiv 2601.16736v5, ICLR 2026) — RESOLVED from UNVERIFIED

**Source (OBSERVED — arXiv API metadata+abstract, fetched 2026-09-29):**
`https://arxiv.org/abs/2601.16736v5` — "A Step to Decouple Optimization in 3DGS",
Renjie Ding, Yaonan Wang, Min Liu, Jialin Zhu, Jiazheng Wang, Jiahao Zhao,
Wenting Shen, Feixiang He, Xiang Chen. Comment: "Accepted by ICLR 2026 (fixed
typo)". Published 2026-01-23. (This is the paper the earlier local summary called
"AdamW-GS"; the name AdamW-GS appears in the abstract as the proposed optimizer.)

**What it does (OBSERVED, abstract):**
- Identifies "two overlooked details in the optimization of 3DGS": "(i) **update step
  coupling**, which induces **optimizer state rescaling** and costly attribute
  updates outside the viewpoints, and (ii) gradient coupling in the moment".
- Decomposes 3DGS optimization into "**Sparse Adam, Re-State Regularization and
  Decoupled Attribute Regularization**", analyzes each empirically under 3DGS and
  3DGS-MCMC, then "re-design[s] the optimization and propose[s] **AdamW-GS by
  re-coupling the beneficial components**, under which better optimization
  efficiency and representation effectiveness are achieved simultaneously".

**Answers (abstract-level):** (a) its Sparse-Adam component skips out-of-view
updates (the "costly attribute updates outside the viewpoints" phrasing); (b) it
does NOT preserve stock dense-Adam semantics — the whole point is to *re-design*
the optimizer (decouple + re-couple into AdamW-GS, a modified optimizer);
(c) NO deferral/replay of skipped transitions — "optimizer state rescaling" is a
state-adjustment heuristic, not exact catch-up of the skipped interval (no replay
claim in the abstract; full-text check still pending, see Section 15);
(d) per-row logical time: **UNVERIFIED at abstract level** — the step-decoupling
analysis implies per-Gaussian step accounting similar in kind to BOGausS's
lifespan, but the exact mechanism needs the full text.

**Full-text check (OBSERVED, 2026-09-29, arxiv.org/html/2601.16736v5, 1.15 MB
fetched):** term counts across the full HTML: `replay` **0**, `catch-up`/`catch
up` **0**, `lazy` **0**, `last.step` **0**, `step counter` **0**, `materiali*`
**0**; the single `deferred` match is arxiv's own banner JavaScript, not paper
content. The paper *names* the dense-Adam zero-gradient behavior VOS preserves —
"**Implicit Update**": "even though [parameters] receive zero gradients, they are
still included in the update step, during which their moments are rescaled and
attributes are still updated" — and treats it as a *defect to redesign away*
("update step coupling ... induces optimizer state rescaling and costly attribute
updates outside the viewpoints"). AdamW-GS's Sparse Adam component therefore
*removes* Implicit Updates for out-of-view Gaussians (the skip family), and its
re-coupled optimizer changes the math. **Confirmed: no replay, no per-row
catch-up, no dense-equivalence claim — NOT a collision.**

**VOS vs AdamW-GS delta:** AdamW-GS belongs to the *modify-the-optimizer* family
(like BOGausS): it changes the update rule and claims better quality/efficiency
from the change. VOS claims the opposite — the trajectory is *identical* to stock
dense Adam, and the gain is purely systems (memory traffic). No collision on the
exact-replay axis; AdamW-GS must be cited as adjacent art on per-Gaussian step
decoupling and as evidence that "update step coupling" is a recognized problem.

### 11.6 SkipGS (arXiv 2603.08997v2) — adjacent hit from the sweep, not a collision

**Source (OBSERVED — arXiv API metadata+abstract, fetched 2026-09-29):** "SkipGS:
Post-Densification Backward Skipping for Efficient 3DGS Training", Jingxing Li,
Yongjae Lee, Deliang Fan. Code: github.com/ASU-ESIC-FAN-Lab/SkipGS.

**What it does (OBSERVED, abstract):** view-adaptive **backward-pass gating** —
"selectively skips backward passes when the sampled view's loss is consistent with
its recent per-view baseline, while enforcing a minimum backward budget". −23.1%
end-to-end training time on Mip-NeRF 360, "comparable" quality.

**Answers:** (a) skips whole backward passes (execution-level skip), not optimizer
rows; (b) does NOT preserve dense-Adam semantics (fewer updates happen at all —
the trajectory changes); (c) no deferral/replay; (d) no per-row logical time.
Different family from VOS (training-loop scheduling vs optimizer-state
virtualization); not a collision. Cite only as context for "skip-style" speedups
changing the trajectory.

---

## 12. Other local prior-art notes (context)

- `p11_opportunity_audit\p11_opportunity_audit\P11_PRIOR_ART_NOTES.md` lines 11-15
  (OBSERVED): "Current Graphdeco also exposes an accelerated rasterizer and
  SparseGaussianAdam. Faster-GS and other engines include fused optimizer code.
  **Generic fused Adam or 'update only visible Gaussians' is therefore not a safe
  novelty direction**, and sparse Adam can differ from dense Adam semantics."
- `baseline_closure\BASELINE_FREEZE.json` line 163 (OBSERVED): explicit statement
  that SparseAdam ≠ dense Adam (quoted in Section 5).
- `candidate_c_source_audit\reports\phase-r2-attribute-decoupled-backward-gate.md`
  lines 260-264 (OBSERVED): local summary of AdamW-GS: "Decouples
  optimizer/update/regularization semantics. Does NOT condition execution of
  different backward derivative families on different Gaussian active sets. It
  modifies the optimizer, not the backward computation graph." — this is the only
  local trace of the AdamW-GS paper; **it is a second-hand summary, not the paper
  itself** (UNVERIFIED upstream).
- `experiments\training_solver_design\topology_sources\gray.html` (OBSERVED,
  in-repo paper snapshot): GRay (Ray Tracing 3D Gaussians) evaluates a
  "SparseAdam" ablation for its ray-traced Gaussian representation (Tab. 9) —
  i.e. skip-invisible-row Adam is applied to non-splatting 3DGS-adjacent systems too.
- `experiments\training_solver_design\topology_sources\augmented_radiance_field_2602_19916v1.html`
  (OBSERVED): Augmented Radiance Field (arXiv 2602.19916v1) uses "SparseAdam ...
  exclusively applied to update the opacity of visible original Gaussian
  primitives" — another skip-family usage in 2D-GS.

---

## 13. Comparison table

| Method | Source | Inactive/zero-grad row handling | Dense-Adam-equivalent? | Deferred transitions replayed? | Per-row logical time? | Exactness claim |
|---|---|---|---|---|---|---|
| torch.optim.Adam (dense) | local torch 2.13 `optim/adam.py` | updates ALL rows (decay + move) | **it is the reference** | n/a | no (global step) | reference semantics |
| torch.optim.SparseAdam | local torch 2.13 `optim/sparse_adam.py`, `_functional.py` L62-84 | **skips** un-touched rows: no decay, no move, no bias corr | NO (docstring: "approximates") | NO | NO (global step) | documented approximation |
| TF LazyAdam | — | UNVERIFIED (best-guess: skip) | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| graphdeco SparseGaussianAdam (accel rasterizer) | `taming-3dgs-rasterizer/submodules/.../cuda_rasterizer/adam.cu` L26-37; `__init__.py` L236-263; official `train.py`/`gaussian_model.py` (fetched) | **skips** invisible rows (`tiles_touched`): no decay, no move; **no bias correction even for visible rows** | NO (README: "can change the training behaviour"; freeze doc line 163) | NO | NO (`state['step']` unused by kernel) | not claimed |
| gsplat SelectiveAdam | `public_baselines/gsplat/gsplat/optimizers/selective_adam.py` L21-101; `cuda/csrc/AdamCUDA.cu` L49-59 | **skips** `valid==false` rows: no decay, no move; no bias correction | NO | NO | NO | not claimed |
| Faster-GS FusedAdam (main) | local `faster-gs/.../torch_bindings/adam.py` L6-36, `adam/src/adam.cu` L10-34; paper §3.3 | updates ALL rows (dense) | YES — "precisely match the behavior of the PyTorch implementation" (paper) + bias correction in kernel | NO | NO (global `state['step']`) | author-claimed exact match |
| Faster-GS "skip invisible" variant | paper §3.3 + Table 3 (fetched); local notes VISIBILITY_LAZY_ADAM_OPPORTUNITY.md L217-235 | **skips** invisible rows, no decay | NO — author: "can cause inconsistencies and performance regressions"; Table 3: −0.14 dB | NO (rows stay frozen forever) | NO | not claimed |
| FastGS `default` | `public_baselines/fastgs/scene/gaussian_model.py` L198-244 | updates all rows; SH on slower clock (every 1/32/64 steps) | YES per group (stock torch Adam incl. bias corr) | NO | NO | inherits torch |
| FastGS `sparse_adam` | same file L211; `train.py` L164-167 | **skips** invisible rows (`radii>0` mask) | NO | NO | NO | not claimed |
| Speedy-Splat | `public_baselines/speedy-splat/train.py` L195-196 | dense Adam, all rows | YES | NO | NO | inherits torch |
| HiGS masked Adam (round-60) | `benchmark/higs_masked_adam.py` (full) | **freezes** masked rows: no decay, no move; **bias correction present** for updated rows (global step) | NO for masked rows ("changes dynamics vs stock Adam", L13-17); bit-close to fused Adam for updated rows (1-2 ulp) | NO | NO (global `state["step"]`) | measured 1-2 ulp vs torch fused for updated rows; frozen rows bit-identical |
| BOGausS (EUSIPCO 2025) | local manuscript `experiments/training_solver_design/topology_sources/bogauss.txt` §IV (OBSERVED) | **skips** un-touched rows (Sparse Adam, as in Taming [4]); modified-Adam math | NO — deliberately a *different* (modified) Adam: `α_{s,t}=max(1/t,1−β*)` blend + precision-scaled steps | NO (no catch-up of skipped interval; grep of full manuscript: zero replay/deferred matches) | **YES — per-Gaussian `t` ("lifespan"), incremented only on update, used for bias correction; inherited α_t·t at splits** | author-claimed "unbiased states" for its own modified estimator, not dense-Adam equivalence |
| AdamW-GS ("A Step to Decouple Optimization in 3DGS", arXiv 2601.16736v5, ICLR 2026) | arXiv abstract + **full text** (Section 11.5) | Sparse-Adam component *removes* "Implicit Updates" (zero-grad moment rescale + attribute move) for out-of-view Gaussians | NO — deliberately re-designs the optimizer (decouple + "re-couple the beneficial components" into AdamW-GS) | NO (full text: zero matches for replay/catch-up/lazy/materialize; "Implicit Update" is removed, not replayed) | NO per-row clock found (full-text terms absent; step-decoupling is analysis-level) | claims better optimization/representation from the *changed* rule |
| SkipGS (arXiv 2603.08997v2) | arXiv API abstract (Section 11.6) | skips whole *backward passes* on plateaued views (not optimizer rows) | NO (fewer updates happen at all) | NO | NO | claims "comparable" quality, not equivalence |
| **lazy-Adam oracle (THIS REPO)** | `experiments/training_solver_design/lazy_adam_oracle.py` + `.json`; `VISIBILITY_LAZY_ADAM_OPPORTUNITY.md` | **defers** frozen rows, **replays** every skipped zero-grad step with exact per-step lr + bias correction before next update; final flush | YES — 4.44e-16 max param error vs torch Adam (CPU FP64, time-varying lr) | **YES** | **YES** (`last_step[i]` per coordinate) | proven numerically in oracle scope; GPU/topology/full-training NOT demonstrated |
| VOS-GS (idea) | task brief | defers per row+attribute, replay on next consumer read | **claimed** YES (exact replay) | **claimed** YES | **claimed** YES (`tau`) | claimed, unimplemented |

---

## 14. COLLISION CHECK

**Question:** does ANY known work implement "per-row timestamp + exact replay of
skipped dense Adam transitions before the next parameter consumer"?

### 14.1 Public prior art

After inspecting every method in Sections 1-11 and searching (to the extent the
network permitted) for the exact-collision phrasings, the closest public works are:

1. **Faster-GS "skip invisible" variant** (arXiv 2602.09999 §3.3, Table 3) — closest
   in *motivation* (skip invisible rows to save time), but it is the **opposite**
   mechanism: it freezes rows permanently and **accepts** the changed trajectory
   (−0.14 dB). No timestamp, no replay, no exactness. The paper itself frames the
   skip as an *optional, lossy* extension.
2. **graphdeco SparseGaussianAdam / gsplat SelectiveAdam / FastGS sparse_adam /
   HiGS masked Adam** — same skip family: freeze or skip, no replay, no timestamp.
   HiGS masked Adam is the most faithful to dense math *for the rows it does update*
   (bias correction included, 1-2 ulp) but still freezes un-updated rows.
3. **BOGausS** (EUSIPCO 2025, DOI 10.23919/EUSIPCO63237.2025.11226626) — **NOW
   VERIFIED (Section 11): full accepted manuscript read from local repo.** It
   maintains a per-Gaussian step counter `t` ("Gaussian's lifespan"), incremented
   only when the Gaussian is updated, used for bias correction in a *modified*
   Sparse-Adam estimator, and inherits that counter at splits (α_t=1, α_g=0.2).
   It does **not** replay deferred transitions and does **not** preserve stock
   dense-Adam semantics. This is the single closest public prior art; the full
   delta table is in Section 11.
4. **Sparse embedding optimizers (PyTorch SparseAdam, TF LazyAdam, RowwiseAdagrad,
   FBGEMM, HugeCTR)** — all in the skip family for un-touched rows (SparseAdam
   verified in Section 2; the rest UNVERIFIED in this audit). None maintains a
   per-row logical clock that is later replayed: the rows simply stay at their last
   touched state, with a *global* step counter used for bias correction. This is a
   **structural** difference from VOS, not a phrasing accident: embedding optimizers
   have no notion of a "parameter consumer" that could trigger replay, because
   every touched row is consumed immediately in the same step.
5. **AdamW-GS** — UNVERIFIED (not found on OpenReview v1 API, HF papers, GitHub
   code search; the only local trace is a second-hand one-line summary in
   candidate_c_source_audit claiming it "decouples optimizer/update/regularization
   semantics" at the parameter-group level). Even if it exists as described, a
   per-parameter-group decoupling is not per-row deferral+replay.

**No public work found that implements per-row timestamps with *exact replay of
skipped stock dense-Adam transitions before the next consumer*.** BOGausS has the
per-Gaussian timestamp but a different (modified) optimizer and no replay; every
other public method in this audit has neither replay nor (in most cases) a
per-row clock.

### 14.2 Internal prior art (same repo, same research program)

**The VOS mechanism itself was already prototyped in this repo** —
`experiments/training_solver_design/lazy_adam_oracle.py` (Section 10) implements
exactly: per-row `last_step[i]` timestamp, deferred zero-gradient dense-Adam
transitions, exact replay (with per-step hyperparameters and bias correction) before
the next update, final flush, validated to 4.44e-16 vs torch Adam. The design doc
`VISIBILITY_LAZY_ADAM_OPPORTUNITY.md` additionally specifies the consumer-driven
materialization points (render, topology events, export) and the displacement-bound
gate. This is internal work, not public prior art — but it means **VOS is not novel
within this project either**: it is the repackaged name for the already-prototyped
"exact deferred dense-Adam settlement" candidate, which was *paused* (not dropped
for lack of opportunity? — see NEXT.md lines 700-709: the opportunity gate FAILED:
even erasing all Adam time saves ≤18.6 s against a 29.06 s 35K-vs-30K wall gap, and
the conservative-visibility gate was still unresolved).

### 14.3 Verdict

**NO_COLLISION (partial)** in public prior art, **with three material caveats**:

- **Caveat 1 (search gap — NARROWED 2026-09-29).** The exact-phrase collision sweep
  was executed on the **arXiv Atom API** (Section 0.1): 13 exact-phrase/keyword
  queries returned **0 relevant results**, and the 2 adjacent hits (AdamW-GS,
  SkipGS) were resolved as non-collisions (Sections 11.5/11.6). Still open:
  (i) arXiv **full-text** search (the API covers title/abstract/metadata only),
  (ii) Semantic Scholar (persistent HTTP 429 from this network), (iii) Google
  Scholar / generic web exact-phrase sweep (engines unreachable; the session
  `web_search` tool is broken with an auth error), (iv) non-arXiv venues
  (EUSIPCO-class conferences, journals, workshop PDFs). The negative result is
  therefore scoped to: every source actually fetched and read (Sections 1-11.6),
  the repo's own prior-art notes, and the arXiv metadata sweep. A full-text +
  Scholar sweep on a working engine remains required before a publication-grade
  novelty claim.
- **Caveat 2 (BOGausS — RESOLVED, delta narrowed).** BOGausS (EUSIPCO 2025) is now
  verified from the full local manuscript (Section 11): it has the per-Gaussian
  step counter ("lifespan") but uses it inside a *modified* sparse-Adam estimator,
  with no replay of skipped transitions and no dense-Adam-equivalence claim. The
  VOS delta vs BOGausS is therefore: (i) exact replay of the *stock* dense-Adam
  recurrence vs BOGausS's modified estimator; (ii) consumer-triggered materialization
  vs BOGausS's update-time-only handling; (iii) VOS's topology flush rules are still
  open while BOGausS ships an explicit (α_t=1, α_g=0.2) inheritance rule. The
  "per-row logical time" component of VOS is **not** novel against BOGausS — the
  local repo's own earlier audits already recorded this (SIBLING_COORDINATES_DESIGN.md
  line 14; NEXT.md lines 1052-1054).
- **Caveat 3 (internal duplication).** Within this repo the exact VOS mechanism is
  already implemented and validated (Section 10). Any novelty claim must be phrased
  against *public* art, and the internal opportunity gate for this mechanism already
  FAILED on measured economics (Section 10; NEXT.md lines 700-709) — the binding
  constraint on the idea is measured headroom, not prior art.

**VOS delta vs each closest prior work:**
- vs **BOGausS** (closest public art, Section 11): VOS replays the *stock*
  dense-Adam recurrence exactly before the next consumer (4.44e-16 in the oracle);
  BOGausS never replays — it runs a *modified* Adam (`α_{s,t} = max(1/t, 1−β*)`
  blend, precision-scaled steps) and claims unbiasedness of its *own* estimator.
  Both have per-Gaussian step counters; VOS's counter drives exact replay,
  BOGausS's drives bias correction of a different recurrence. Both handle
  topology; BOGausS ships an explicit inheritance rule, VOS's flush/ordering rules
  are still open.
- vs Faster-GS skip-invisible: VOS does not accept the changed trajectory — it
  replays the frozen interval exactly. Faster-GS skip is a deliberate approximation
  (−0.14 dB, author-acknowledged).
- vs SparseGaussianAdam/SelectiveAdam/HiGS masked-Adam: those omit decay entirely
  (and the 3DGS ones also omit bias correction); VOS restores the exact dense
  recurrence.
- vs SparseAdam/LazyAdam/embedding optimizers: those are for sparse-grad-layout
  embeddings with no consumer-driven materialization; none maintains a per-row
  logical clock that is later replayed (global step counter + skip).
- vs the internal oracle (Section 10): **no delta** — VOS *is* the oracle's
  mechanism; the open deltas are GPU implementation, the conservative-visibility
  gate (screen-envelope proxy, one 272-camera sample only), topology-event flush
  semantics (clone/split/prune/SH-degree/checkpoint ordering, per
  newborn_adam_audit.py), and full-training E2E + quality evidence.

---

## 15. Recommendations (audit scope only, no promotion)

1. ~~Execute the exact-phrase sweep~~ — **DONE on the arXiv API** (Section 0.1;
   clean). Remaining: arXiv full-text, Semantic Scholar (429-blocked), Google
   Scholar / generic web (unreachable from this network; `web_search` tool broken).
   Priority phrases for the remaining engines: "deferred optimizer state
   materialization", "per-row timestamp optimizer", "virtual optimizer state",
   "logical optimizer step replay", "exact replay skipped Adam updates", "Adam
   moment decay lazy evaluation", "momentum decay for inactive parameters exact
   equivalence", "sparse Adam dense equivalence bias correction".
2. **BOGausS is now verified (Section 11)** — no further fetch needed; its delta
   vs VOS is documented. If a publication claim is to be made, the per-Gaussian
   timestamp must be attributed to BOGausS, and novelty must be scoped to
   (i) exact replay of stock dense-Adam transitions, (ii) consumer-triggered
   materialization, and (iii) topology-flush semantics.
3. Verify the Faster-GS `FasterGSFused` branch (backward/optimizer fusion) source —
   the local mirror only contains the `main` branch FusedAdam.
4. Locate the TF LazyAdam source module (not in `optimizer_v2/adam.py` on master)
   and record its row-skip semantics precisely.
5. ~~Attempt to locate AdamW-GS~~ — **RESOLVED and full-text-checked** (Section
   11.5): arXiv 2601.16736v5, "A Step to Decouple Optimization in 3DGS", ICLR
   2026. Full-text term sweep: zero occurrences of replay/catch-up/lazy/
   materialize; its "Implicit Update" concept names the exact dense-Adam
   zero-gradient behavior but is *removed*, not replayed. No collision.
6. Note for the record: the internal opportunity gate for exactly this mechanism
   already FAILED on measured economics (≤18.6 s reclaimable Adam time vs 29.06 s
   35K-vs-30K wall gap, plus unresolved visibility gate) —
   `VISIBILITY_LAZY_ADAM_OPPORTUNITY.md` lines 105-120 and `NEXT.md` lines 700-709.
   Prior art is not the binding constraint on this idea; the measured headroom is.
