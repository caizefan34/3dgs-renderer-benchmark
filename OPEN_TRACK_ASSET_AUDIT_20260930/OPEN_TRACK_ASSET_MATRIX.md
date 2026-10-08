# Open-Track Frozen Asset Audit — 2026-09-30

- Basis: **LIVE EPIC-05 verification** (SSH sha256sum). Supersedes PRELIMINARY_MATRIX.json (generated while remote was offline).
- Authoritative host: EPIC-05 (dsw-933622-64cc845c7d-lqswc), 8× NVIDIA A100-SXM4-80GB, torch 2.9.1+cu128, CUDA 12.8.
- Audit UTC: 2026-09-30T02:32:33Z
- No scientific verdict issued. No experiments run. No checkpoints regenerated. No frozen binaries modified.

## 1. Frozen binary identity (all MATCH authoritative pins)

| binary | expected sha256 | found sha256 | match |
|---|---|---|---|
| core gsplat_cuda.so (frozen C0 lineage) | `935ea6c0dd04…` | `935ea6c0dd04…` | MATCH |
| C0 extension .so (P12 publication lineage) | `d26cccf46c79…` | `d26cccf46c79…` | MATCH |
| fused_ssim_cuda.so (P11-locked loss backend) | `dd6a4c799b98…` | `dd6a4c799b98…` | MATCH |
| P12 publication trainer | `675b0267f3a1…` | `675b0267f3a1…` | MATCH |
| publication trainer (AB0 lineage) | `1b67171d93bc…` | `1b67171d93bc…` | MATCH |
| gaussian_model.py (reference_v1) | `68731e375013…` | `68731e375013…` | MATCH |
| config.py (reference_v1) | `ea8e870476a3…` | `ea8e870476a3…` | MATCH |
| colmap_reader.py (reference_v1) | `b30a2a345f7f…` | `b30a2a345f7f…` | MATCH |

Core gsplat_cuda.so is 90,249,072 bytes (fully built, not a stub). C0 extension = d26cccf…, core = 935ea6c0….

## 2. Canonical checkpoint inventory (C0 lineage, provenance-verified)

| scene | iter | classification | sha256 (12) | provenance |
|---|---|---|---|---|
| room | 10000K | CANONICAL_VERIFIED | `9aa00f0ddf28…` | CVT0_SOURCE_IDENTITY.json, MG0_CHECKPOINT_IDENTITY_room.json, N1_CHECKPOINT_IDENTITY_room.json, N1D_SOURCE_IDENTITY.json |
| room | 15000K | CANONICAL_VERIFIED | `b276b338f5a4…` | CVT0_SOURCE_IDENTITY.json, MG0_CHECKPOINT_IDENTITY_room.json, N1_CHECKPOINT_IDENTITY_room.json, N1D_SOURCE_IDENTITY.json |
| room | 20000K | CANONICAL_VERIFIED | `c4e41cfef79e…` | CVT0_SOURCE_IDENTITY.json, MG0_CHECKPOINT_IDENTITY_room.json, N1_CHECKPOINT_IDENTITY_room.json, N1D_SOURCE_IDENTITY.json |
| room | 25000K | CANONICAL_VERIFIED | `b31e5a6b0e73…` | A0_SOURCE_IDENTITY.json, MG0_CHECKPOINT_IDENTITY_room.json, N1_CHECKPOINT_IDENTITY_room.json, N1D_SOURCE_IDENTITY.json |
| bicycle | 10000K | CANONICAL_VERIFIED | `26bf0b959f10…` | CVT0_SOURCE_IDENTITY.json, MG0_CHECKPOINT_IDENTITY_bicycle.json, N1_CHECKPOINT_IDENTITY_bicycle.json, N1D_SOURCE_IDENTITY.json |
| bicycle | 15000K | CANONICAL_VERIFIED | `273971d3867c…` | CVT0_SOURCE_IDENTITY.json, MG0_CHECKPOINT_IDENTITY_bicycle.json, N1_CHECKPOINT_IDENTITY_bicycle.json, N1D_SOURCE_IDENTITY.json |
| bicycle | 20000K | CANONICAL_VERIFIED | `96f136cb331a…` | CVT0_SOURCE_IDENTITY.json, MG0_CHECKPOINT_IDENTITY_bicycle.json, N1_CHECKPOINT_IDENTITY_bicycle.json, N1D_SOURCE_IDENTITY.json |
| bicycle | 25000K | CANONICAL_VERIFIED | `73ae5436cc44…` | MG0_CHECKPOINT_IDENTITY_bicycle.json, N1_CHECKPOINT_IDENTITY_bicycle.json, N1D_SOURCE_IDENTITY.json |
| room | 5000K | REGENERATED_NONCANONICAL | `4295f6df748c…` | H0_R5K_IDENTITY.json (regen_sha256; suffix identity gate MISMATCH) |
| bicycle | 5000K | REGENERATED_NONCANONICAL | `f41709dab8f9…` | H0_R5K_IDENTITY.json (regen_sha256; suffix identity gate MISMATCH) |
| room | 30000K | AUDIT_ONLY_UNPINNED | `9796da5b99f5…` | (no independent SOURCE_IDENTITY pins this hash) |
| bicycle | 30000K | AUDIT_ONLY_UNPINNED | `45f5440cc778…` | (no independent SOURCE_IDENTITY pins this hash) |

Classification key: CANONICAL_VERIFIED (physically present + SHA256 matches independent provenance pin); REGENERATED_NONCANONICAL (H0-R5K regen, identity gate FAILED); AUDIT_ONLY_UNPINNED (full30k c0 30K, present but no independent pin).

## 3. Track asset matrix

| track | existing scientific status | asset_status | missing assets | permitted next action |
|---|---|---|---|---|
| **CVT0** | CVT0_DROP (6/6 cells executed; promote=false strong=false on all room/bicycle @10K/15K/20K) | **COMPLETED** | — | None. CVT0 is a completed scientific direction with a DROP verdict. Assets exist and are verified, but no rerun is authorized (frozen prereg already consumed). |
| **IG0** | CPU_REFERENCE_PASS (6/6 fp64 correctness controls PASS); GPU scientific probes NOT_MEASURED (IG0_STRONG_PROMOTE/PROMOTE/DROP not issued) | **RUNNABLE** | — | Assets exist and identity gate passes on remote. GPU probes (IG0_PROBES.json / IG0_TRUE_HOLDOUT.json) remain to be executed under the existing IG0 prereg; RUNNABLE does not itself authorize a verdict. |
| **S0** | IN_PROGRESS (SecantGS S0): 1/expected cells complete (room@10000_off0.done); no S0 verdict/summary artifact present yet | **RUNNABLE** | — | Assets exist and are verified; S0 execution already in progress. Continue S0 under existing frozen S0_PREREG.md (do not change gates/checkpoints). |
| **ET0** | NOT_EXECUTED (no ET0 verdict/report/prereg found on EPIC-05; only et0_cell.py/et0_core.py scripts and a smoke env json exist) | **RUNNABLE** | — | Assets exist and are verified, but NO ET0 prereg exists. A new preregistration is REQUIRED before any scientific ET0 execution. Do not execute against an unpublished protocol. |
| **EM0** | EM0_BLOCKED_MISSING_FROZEN_INPUTS (frozen by user directive); verdict NOT_MEASURED | **BLOCKED_MISSING_FROZEN_INPUTS** | EM0_DIRECTIONS.json, EM0_SOURCE_IDENTITY.json | Do NOT reopen unless all required historical canonical ET0 artifacts actually exist. They do not. Remains blocked. |
| **CR0** | NOT_EXECUTED (no CR0 directory, verdict, or prereg found on EPIC-05) | **BLOCKED_MISSING_FROZEN_INPUTS** | ET0-generated traces | Blocked upstream: ET0 traces absent and no CR0 prereg exists. Do not execute; requires ET0 completion + prereg. |
| **H0** | H0_CHECKPOINT_BLOCKER (original prereg verdict; missing canonical 5K cells) | **BLOCKED_MISSING_FROZEN_INPUTS** | room@5K, bicycle@5K | Original H0 6-cell grid cannot run: canonical 5K does not exist and must NOT be reconstructed (INFRA-C0-NONDETERMINISM). Superseded for the available cells by H0-FROZEN prereg. Do not rerun original H0. |
| **H0-R5K** | H0_R5K_REGEN_IDENTITY_FAIL (regenerated 10K/15K/20K/25K MISMATCH canonical 8/8; new 5K suffix identity gate failed) | **CLOSED** | bit-exact canonical 5K (UNSATISFIABLE: cross-run non-determinism) | IMMUTABLE. Do NOT rerun as a rescue. Not a HomotopyGS scientific DROP; it is an infrastructure reproducibility blocker. Any path assuming bit-exact frozen-trajectory regeneration is forbidden without explicit research-lead sign-off to change frozen binaries/relax prereg gates. |
| **H0-FROZEN** | PROPOSED prereg (H0_FROZEN_PREREG.md present; NOT yet executed; no H0_FROZEN verdict) | **RUNNABLE** | — | ASSET-RUNNABLE: all 4 required canonical cells exist and are independently verified. This is NOT an H0 verdict. Execution requires the existing H0_FROZEN prereg to be run on EPIC-05 (reduced 4-cell cohort, gates unchanged). |
| **AB0** | AB0_CHECKPOINT_BLOCKER (identity_gate PASS, checkpoint_gate FAIL; 0 experiments run) | **BLOCKED_MISSING_FROZEN_INPUTS** | room@5K, bicycle@5K | Blocked on canonical 5K. No substitute early checkpoint permitted. Requires research-lead decision to authorize frozen-C0 retraining (deviation from exact-checkpoint gate) or keep blocked. 15K is no longer the blocker. |
| **SG0** | SG0_BASELINE_UNSTABLE (original 20K/25K experiment completed; >2/8 BASE_NON_DEScent for K=8 -> STOP) | **COMPLETED** | — | None. SG0 is COMPLETED with SG0_BASELINE_UNSTABLE. Do NOT rerun. (Canonical room/bicycle 10K/15K/20K/25K also exist and are verified, but that does not reopen SG0.) |
| **SG0-E10** | PROPOSED (not original SG0-Early; NO prereg yet; not executed) | **RUNNABLE** | — | ASSET-RUNNABLE: canonical room/bicycle 10K exist and are verified. Requires a NEW preregistration before scientific execution (explicitly not original SG0-Early). No verdict issued by this audit. |

### Per-track detail

#### CVT0
- scientific_status: CVT0_DROP (6/6 cells executed; promote=false strong=false on all room/bicycle @10K/15K/20K)
- required_assets: room@10K; room@15K; room@20K; bicycle@10K; bicycle@15K; bicycle@20K (exact frozen C0)
- found_assets:
  - room@10000K=CANONICAL_VERIFIED:9aa00f0ddf28
  - room@15000K=CANONICAL_VERIFIED:b276b338f5a4
  - room@20000K=CANONICAL_VERIFIED:c4e41cfef79e
  - bicycle@10000K=CANONICAL_VERIFIED:26bf0b959f10
  - bicycle@15000K=CANONICAL_VERIFIED:273971d3867c
  - bicycle@20000K=CANONICAL_VERIFIED:96f136cb331a
- missing_assets: none
- provenance_conflicts:
- asset_status: **COMPLETED**
- permitted_next_action: None. CVT0 is a completed scientific direction with a DROP verdict. Assets exist and are verified, but no rerun is authorized (frozen prereg already consumed).

#### IG0
- scientific_status: CPU_REFERENCE_PASS (6/6 fp64 correctness controls PASS); GPU scientific probes NOT_MEASURED (IG0_STRONG_PROMOTE/PROMOTE/DROP not issued)
- required_assets: room@10K; room@15K; bicycle@10K; bicycle@15K (exact frozen C0) for GPU stage
- found_assets:
  - room@10000K=CANONICAL_VERIFIED:9aa00f0ddf28
  - room@15000K=CANONICAL_VERIFIED:b276b338f5a4
  - bicycle@10000K=CANONICAL_VERIFIED:26bf0b959f10
  - bicycle@15000K=CANONICAL_VERIFIED:273971d3867c
- missing_assets: none
- provenance_conflicts:
  - Preliminary matrix (offline) marked IG0 GPU PRELIM_BLOCKED_GPU canonical 10K/15K absent; LIVE EPIC-05 verification shows canonical 10K/15K present and hash-matched. Offline evidence SUPERSEDED.
- asset_status: **RUNNABLE**
- permitted_next_action: Assets exist and identity gate passes on remote. GPU probes (IG0_PROBES.json / IG0_TRUE_HOLDOUT.json) remain to be executed under the existing IG0 prereg; RUNNABLE does not itself authorize a verdict.

#### S0
- scientific_status: IN_PROGRESS (SecantGS S0): 1/expected cells complete (room@10000_off0.done); no S0 verdict/summary artifact present yet
- required_assets: room@10K; room@15K; bicycle@10K; bicycle@15K (exact frozen C0)
- found_assets:
  - room@10000K=CANONICAL_VERIFIED:9aa00f0ddf28
  - room@15000K=CANONICAL_VERIFIED:b276b338f5a4
  - bicycle@10000K=CANONICAL_VERIFIED:26bf0b959f10
  - bicycle@15000K=CANONICAL_VERIFIED:273971d3867c
- missing_assets: none
- provenance_conflicts:
  - Preliminary matrix (offline) marked S0 PRELIM_BLOCKED canonical 10K/15K absent; LIVE EPIC-05 verification shows canonical 10K/15K present and hash-matched. Offline evidence SUPERSEDED. S0_PLAN identity gate already recorded PASS on remote.
- asset_status: **RUNNABLE**
- permitted_next_action: Assets exist and are verified; S0 execution already in progress. Continue S0 under existing frozen S0_PREREG.md (do not change gates/checkpoints).

#### ET0
- scientific_status: NOT_EXECUTED (no ET0 verdict/report/prereg found on EPIC-05; only et0_cell.py/et0_core.py scripts and a smoke env json exist)
- required_assets: room@10K; room@15K; room@20K; bicycle@10K; bicycle@15K; bicycle@20K (exact frozen C0)
- found_assets:
  - room@10000K=CANONICAL_VERIFIED:9aa00f0ddf28
  - room@15000K=CANONICAL_VERIFIED:b276b338f5a4
  - room@20000K=CANONICAL_VERIFIED:c4e41cfef79e
  - bicycle@10000K=CANONICAL_VERIFIED:26bf0b959f10
  - bicycle@15000K=CANONICAL_VERIFIED:273971d3867c
  - bicycle@20000K=CANONICAL_VERIFIED:96f136cb331a
- missing_assets: none
- provenance_conflicts:
  - Preliminary matrix (offline) marked ET0 PRELIM_BLOCKED canonical 10K/15K/20K, 10K/15K absent; LIVE EPIC-05 verification shows canonical 10K/15K/20K present and hash-matched. Offline evidence SUPERSEDED.
- asset_status: **RUNNABLE**
- permitted_next_action: Assets exist and are verified, but NO ET0 prereg exists. A new preregistration is REQUIRED before any scientific ET0 execution. Do not execute against an unpublished protocol.

#### EM0
- scientific_status: EM0_BLOCKED_MISSING_FROZEN_INPUTS (frozen by user directive); verdict NOT_MEASURED
- required_assets: EM0_DIRECTIONS.json (ET0-generated); EM0_SOURCE_IDENTITY.json (ET0-generated); room@10K; room@15K; bicycle@10K; bicycle@15K
- found_assets:
  - room@10000K=CANONICAL_VERIFIED:9aa00f0ddf28
  - room@15000K=CANONICAL_VERIFIED:b276b338f5a4
  - bicycle@10000K=CANONICAL_VERIFIED:26bf0b959f10
  - bicycle@15000K=CANONICAL_VERIFIED:273971d3867c
- missing_assets: EM0_DIRECTIONS.json, EM0_SOURCE_IDENTITY.json
- provenance_conflicts:
  - EM0_DIRECTIONS.json and EM0_SOURCE_IDENTITY.json do NOT exist anywhere on EPIC-05 (verified by full-filesystem search). Canonical 10K/15K DO exist (offline matrix wrongly implied they were absent).
  - ET0 has not been executed, so its required historical direction/identity artifacts cannot exist.
- asset_status: **BLOCKED_MISSING_FROZEN_INPUTS**
- permitted_next_action: Do NOT reopen unless all required historical canonical ET0 artifacts actually exist. They do not. Remains blocked.

#### CR0
- scientific_status: NOT_EXECUTED (no CR0 directory, verdict, or prereg found on EPIC-05)
- required_assets: ET0-generated traces; room@10K; room@15K; room@20K; bicycle@10K; bicycle@15K; bicycle@20K
- found_assets:
  - room@10000K=CANONICAL_VERIFIED:9aa00f0ddf28
  - room@15000K=CANONICAL_VERIFIED:b276b338f5a4
  - room@20000K=CANONICAL_VERIFIED:c4e41cfef79e
  - bicycle@10000K=CANONICAL_VERIFIED:26bf0b959f10
  - bicycle@15000K=CANONICAL_VERIFIED:273971d3867c
  - bicycle@20000K=CANONICAL_VERIFIED:96f136cb331a
- missing_assets: ET0-generated traces
- provenance_conflicts:
  - Canonical 10K/15K/20K exist (offline matrix wrongly implied 10K/15K absent).
  - CR0 upstream dependency is ET0-generated traces, which do not exist (ET0 not executed). No CR0 prereg found.
- asset_status: **BLOCKED_MISSING_FROZEN_INPUTS**
- permitted_next_action: Blocked upstream: ET0 traces absent and no CR0 prereg exists. Do not execute; requires ET0 completion + prereg.

#### H0
- scientific_status: H0_CHECKPOINT_BLOCKER (original prereg verdict; missing canonical 5K cells)
- required_assets: room@5K; room@15K; room@25K; bicycle@5K; bicycle@15K; bicycle@25K
- found_assets:
  - room@15000K=CANONICAL_VERIFIED:b276b338f5a4
  - room@25000K=CANONICAL_VERIFIED:b31e5a6b0e73
  - bicycle@15000K=CANONICAL_VERIFIED:273971d3867c
  - bicycle@25000K=CANONICAL_VERIFIED:73ae5436cc44
- missing_assets: room@5K, bicycle@5K
- provenance_conflicts:
  - H0_VERDICT.json (generated offline) listed room/bicycle @15000/@25000 as missing_cells; LIVE EPIC-05 verification shows canonical 15K/25K present and hash-matched. Offline evidence SUPERSEDED for 15K/25K; 5K genuinely ABSENT (only H0-R5K regen + vos_traces other-renderer 5K exist).
- asset_status: **BLOCKED_MISSING_FROZEN_INPUTS**
- permitted_next_action: Original H0 6-cell grid cannot run: canonical 5K does not exist and must NOT be reconstructed (INFRA-C0-NONDETERMINISM). Superseded for the available cells by H0-FROZEN prereg. Do not rerun original H0.

#### H0-R5K
- scientific_status: H0_R5K_REGEN_IDENTITY_FAIL (regenerated 10K/15K/20K/25K MISMATCH canonical 8/8; new 5K suffix identity gate failed)
- required_assets: regenerated 5K that is bit-exact to canonical (unsatisfiable under frozen binaries)
- found_assets:
  - regenerated room@5K=4295f6df748cf (REGENERATED_NONCANONICAL)
  - regenerated bicycle@5K=f41709dab8f9 (REGENERATED_NONCANONICAL)
- missing_assets: bit-exact canonical 5K (UNSATISFIABLE: cross-run non-determinism)
- provenance_conflicts:
  - H0_R5K_IDENTITY.json: all 8 regenerated cells (room/bicycle @10K/15K/20K/25K) MISMATCH canonical SHA256; new 5K regen recorded but identity gate failed.
  - C0 training is cross-run non-deterministic (custom CUDA rasterizer backward atomicAdd); torch determinism APIs cannot control it. Bit-exact regeneration is UNSATISFIABLE.
- asset_status: **CLOSED**
- permitted_next_action: IMMUTABLE. Do NOT rerun as a rescue. Not a HomotopyGS scientific DROP; it is an infrastructure reproducibility blocker. Any path assuming bit-exact frozen-trajectory regeneration is forbidden without explicit research-lead sign-off to change frozen binaries/relax prereg gates.

#### H0-FROZEN
- scientific_status: PROPOSED prereg (H0_FROZEN_PREREG.md present; NOT yet executed; no H0_FROZEN verdict)
- required_assets: room@15K; room@25K; bicycle@15K; bicycle@25K (independently verified canonical)
- found_assets:
  - room@15000K=CANONICAL_VERIFIED:b276b338f5a4
  - room@25000K=CANONICAL_VERIFIED:b31e5a6b0e73
  - bicycle@15000K=CANONICAL_VERIFIED:273971d3867c
  - bicycle@25000K=CANONICAL_VERIFIED:73ae5436cc44
- missing_assets: none
- provenance_conflicts:
  - Preliminary matrix marked H0-FROZEN NEEDS_REMOTE_PROOF evidence inconsistent about 15K; LIVE EPIC-05 verification confirms canonical room/bicycle 15K and 25K physically exist and are SHA256-verified across MG0/N1/N1D/A0 provenance. Evidence RESOLVED in favor of existence.
- asset_status: **RUNNABLE**
- permitted_next_action: ASSET-RUNNABLE: all 4 required canonical cells exist and are independently verified. This is NOT an H0 verdict. Execution requires the existing H0_FROZEN prereg to be run on EPIC-05 (reduced 4-cell cohort, gates unchanged).

#### AB0
- scientific_status: AB0_CHECKPOINT_BLOCKER (identity_gate PASS, checkpoint_gate FAIL; 0 experiments run)
- required_assets: room@5K; room@15K; bicycle@5K; bicycle@15K (exact canonical; no substitution)
- found_assets:
  - room@15000K=CANONICAL_VERIFIED:b276b338f5a4
  - bicycle@15000K=CANONICAL_VERIFIED:273971d3867c
- missing_assets: room@5K, bicycle@5K
- provenance_conflicts:
  - AB0_VERDICT.json (offline) asserted 5K/15K absent; LIVE verification shows canonical 15K present (hash-matched). Offline evidence SUPERSEDED for 15K.
  - Canonical 5K genuinely ABSENT (only D0-TRACE/N1-c0f-15K-only/GRC3D-gmain other-arm 5K, and H0-R5K regen 5K, none substitutable per prereg section 1).
- asset_status: **BLOCKED_MISSING_FROZEN_INPUTS**
- permitted_next_action: Blocked on canonical 5K. No substitute early checkpoint permitted. Requires research-lead decision to authorize frozen-C0 retraining (deviation from exact-checkpoint gate) or keep blocked. 15K is no longer the blocker.

#### SG0
- scientific_status: SG0_BASELINE_UNSTABLE (original 20K/25K experiment completed; >2/8 BASE_NON_DEScent for K=8 -> STOP)
- required_assets: original SG0 20K/25K cells (already consumed)
- found_assets:
  - SG0_VERDICT.json present (executed)
- missing_assets: none
- provenance_conflicts:
- asset_status: **COMPLETED**
- permitted_next_action: None. SG0 is COMPLETED with SG0_BASELINE_UNSTABLE. Do NOT rerun. (Canonical room/bicycle 10K/15K/20K/25K also exist and are verified, but that does not reopen SG0.)

#### SG0-E10
- scientific_status: PROPOSED (not original SG0-Early; NO prereg yet; not executed)
- required_assets: room@10K; bicycle@10K (exact canonical)
- found_assets:
  - room@10000K=CANONICAL_VERIFIED:9aa00f0ddf28
  - bicycle@10000K=CANONICAL_VERIFIED:26bf0b959f10
- missing_assets: none
- provenance_conflicts:
  - Preliminary matrix marked SG0-E10 NEEDS_REMOTE_PROOF canonical 10K absent; LIVE EPIC-05 verification confirms canonical room/bicycle 10K physically exist and are SHA256-verified (CVT0/MG0/N1/N1D provenance). Evidence RESOLVED in favor of existence. This is NOT original SG0-Early.
- asset_status: **RUNNABLE**
- permitted_next_action: ASSET-RUNNABLE: canonical room/bicycle 10K exist and are verified. Requires a NEW preregistration before scientific execution (explicitly not original SG0-Early). No verdict issued by this audit.

## 4. Reproducibility finding (infrastructure, not a scientific DROP)

Same seed/source/binaries/data did NOT reproduce canonical C0 checkpoint bytes. H0-R5K suffix identity gate failed: regenerated room/bicycle 10K/15K/20K/25K MISMATCH canonical 8/8. Cross-run training is not bit-exact reproducible.

Therefore: regenerated checkpoints may not be labeled canonical; missing canonical checkpoints may not be reconstructed after the fact; deterministic camera RNG is not evidence of bit-exact training reproducibility. The atomicAdd non-determinism in the custom CUDA rasterizer backward is a **strongly-supported mechanism**, not claimed as the proven sole cause absent a deterministic-accumulation intervention.

## 5. Provenance conflicts (preliminary vs live)

The shipped PRELIMINARY_MATRIX.json was generated while EPIC-05 was OFFLINE and marked canonical 10K/15K (and H0 15K/25K) as absent. Live verification SUPERSEDES this:

| id | topic | resolution | tracks |
|---|---|---|---|
| PC-01 | canonical 10K/15K existence | SUPERSEDED: assets exist and are canonical-verified | CVT0, IG0, S0, ET0, EM0, CR0 |
| PC-02 | canonical 20K existence | SUPERSEDED: assets exist | ET0, CR0 |
| PC-03 | canonical 15K/25K for H0-FROZEN | RESOLVED: assets exist and are verified | H0-FROZEN |
| PC-04 | canonical 5K existence | CONFIRMED ABSENT: canonical 5K does not exist; must not be reconstructed | H0, AB0, H0-R5K |
| PC-05 | full30k c0 30K provenance | AUDIT_ONLY_UNPINNED: physically present, provenance not independently verified |  |
| PC-06 | cross-run reproducibility | CONFIRMED: C0 training is cross-run non-deterministic; bit-exact regeneration UNSATISFIABLE. atomicAdd non-determinism is a strongly-supported mechanism, not claimed as proven sole cause absent a deterministic-accumulation intervention. | H0-R5K, H0, AB0, SG0-E10 |
| PC-07 | ET0-generated EM0 artifacts | CONFIRMED ABSENT: EM0/CR0 remain blocked on upstream ET0 artifacts | EM0, CR0 |

Canonical **5K remains genuinely ABSENT** (only H0-R5K regen + vos_traces other-renderer 5K). EM0/CR0 remain blocked on absent ET0-generated artifacts.
