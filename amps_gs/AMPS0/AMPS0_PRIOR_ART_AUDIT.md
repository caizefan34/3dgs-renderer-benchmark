# AMPS0 Prior-Art Audit — Attribute-aware Mixed-Precision Optimizer States for Gaussian Splatting

**Generated:** 2026-10-05
**Lane:** AMPS0 (independent; does not modify CGT/DPS0/IGF/RetireGS/ClockGS/Sparse-tail K50-B1)
**Purpose:** Mandatory mechanism-level audit before implementation (objective section 1).
**Novelty classification vocabulary:** `NO_DIRECT_GS_MATCH_FOUND` / `PARTIAL_OVERLAP` / `DIRECT_PRIOR_ART_CONFLICT`

## Verification status (honest disclosure)

- The arxiv export API (`export.arxiv.org`) was **network-blocked** in this execution environment
  (HTTP status 000 / timeouts on every query). `arxiv.org` abstract pages were reachable, but
  title-based search could not be completed programmatically here.
- Therefore the entries below are compiled from **internal knowledge** of these works and cross-checked
  against the local repo evidence (`fgs_strong_baseline_final/`, `baseline_closure/`,
  `momentgrad_gs/`, gsplat 1.4.0 source tree). Each row carries a `verification` field.
- Per objective section 26, a **fresh literature audit is required before any paper claim**. This
  audit establishes the engineering-lane baseline only; it does NOT authorize a novelty claim.

---

## Mechanism-level audit

| # | Work (repo/paper) | Year | Exact optimizer mechanism | Optimizer equation changes? | Moment-state precision changes? | Attribute-aware? | Training speed reported? | Fused w/ backward? | Overlap with AMPS | verification |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Faster-GS** (`nerficg-project/faster-gaussian-splatting`, FasterGSFused) | 2024 | Fused render+backward+Adam CUDA kernel; efficient 3DGS | No | No (FP32 m/v) | No | Yes (3.9x vs basis, 1.45x vs C0 cross-backend) | Yes (fused) | AMPS0 uses it as the strong baseline; it does NOT compress optimizer states | verified-local (repo) |
| 2 | **8-bit Adam** (Dettmers et al., "8-bit Optimizers via Block-wise Quantization") | 2022 | Block-wise dynamic quantization of m/v to 8-bit + block-stable mean-field shift | No | **Yes** (8-bit m/v) | No (uniform across all params) | Yes (LLM training) | No | Closest generic mechanism; NOT attribute-aware, NOT GS-specific, NOT fused with 3DGS backward | internal-knowledge |
| 3 | **bitsandbytes optimizers** (Dettmers et al.) | 2020 | 8-bit Adam via dynamic quantization + LLM.int8() | No | Yes (8-bit) | No | Yes | No | Generic 8-bit optimizer; same as #2 | internal-knowledge |
| 4 | **SparseGaussianAdam / SelectiveAdam** (gsplat `sparse_adam`; also Taming-3DGS) | 2024 | Skip optimizer update for invisible (zero-contribution) Gaussians; no decay for invisible, no bias correction | **Yes** (skips updates, changes bias correction) | No | Partially (visibility, not attribute type) | Yes (2-3x on some scenes) | No (separate step) | **EXCLUDED by objective section 27** (skip-invisible is a different mechanism); overlaps the "skip-invisible" lane, NOT state precision | verified-local (gsplat tree + baseline_closure) |
| 5 | **GS-Offload / GS-Scale** (offloading family) | 2024 | Offload optimizer states / Gaussians to CPU/NVMe, prefetch | No | No (keeps FP32, moves location) | No | Yes (memory-bound wins) | No | Different axis (location, not precision); no overlap with state-precision | internal-knowledge |
| 6 | **AdamW-GS** (decoupled weight decay for GS) | 2024 | AdamW decoupled weight decay on Gaussian params | **Yes** (adds weight decay) | No | No | Yes | No | **EXCLUDED by section 27** (changed optimizer equation) | internal-knowledge |
| 7 | **BOGausS** (block-optimized GS) | 2024 | Block-structured / orthogonal updates for GS | Likely Yes | Unclear | Unclear | Unclear | Unclear | Name suggests block-wise optimization; if it quantizes states it would conflict — needs fresh verification | internal-knowledge (LOW confidence — needs fresh lit audit) |
| 8 | **Generic mixed-precision optimizer-state methods** (FP16 master + mixed precision, Micikevicius et al.; Adafactor; 8-bit RAGO) | 2018-2023 | Mixed-precision / factorized / quantized optimizer states | Sometimes | Yes | No | Sometimes | No | Generic; not GS, not attribute-aware, not fused with 3DGS | internal-knowledge |
| 9 | **HAC / SOG / LightGaussian / E-D3DGS** (GS *compression* family) | 2024 | Quantize/compress the **Gaussian parameters/attributes** (and optionally SH), not the Adam moments | No | Compresses **params**, not optimizer m/v | Yes (attribute-aware param compression!) | Yes (render speed via smaller model) | No (offline compression) | **Attribute-aware compression EXISTS but targets parameters, not optimizer states.** AMPS0 targets m/v state precision specifically — a different object. No direct match found. | internal-knowledge |
| 10 | **Vanilla 3DGS / INRIA Adam** | 2023 | Standard Adam, FP32 m/v, separate step | No | No | No | Yes | No | Reference only | verified-local |

---

## Classification

**Primary mechanism ("geometry FP32 + appearance lower-precision Adam states + fused 3DGS optimizer"): `NO_DIRECT_GS_MATCH_FOUND`.**

Rationale:
- Generic 8-bit / block-wise quantized optimizers (#2, #3, #8) change moment-state precision but are
  **uniform across all parameters** — they do not exploit the 3DGS attribute asymmetry
  (geometry vs. high-dimensional SH) that AMPS0 targets.
- GS *compression* methods (#9) are attribute-aware but compress **parameters**, not the **optimizer
  moment states** (m/v). The compressed object is different.
- Sparse/Selective Adam (#4) skips updates by visibility — a different mechanism, explicitly excluded
  by section 27.
- Offloading (#5) moves states, does not reduce their precision.
- No audited work implements "keep geometry m/v FP32, store high-dimensional SH-rest m/v in BF16,
  compute in FP32, fused into the 3DGS backward/optimizer, lifecycle-aware under densification."

**Partial overlaps (must be cited, not claimed over):**
- 8-bit/block-wise quantization of optimizer states (#2, #3, #8) — the storage-precision *technique* is not new.
- Attribute-aware compression in GS (#9) — the *attribute-asymmetry insight* is not new, but it is applied to parameters, not optimizer states.

**Novelty boundary (per section 26):** AMPS0 novelty, if any, must be **mechanism-specific**:
1. exploiting 3DGS attribute asymmetry to split optimizer-state precision (geometry FP32 / SH BF16),
2. reducing high-dimensional **appearance-state** traffic specifically,
3. integrating into the **fused** Gaussian backward/optimizer,
4. lifecycle-aware precision under dynamic densification (clone/split/prune/reset/resize).

Do NOT claim "first mixed-precision Adam" or "first compressed optimizer."

**Caveat:** BOGausS (#7) is LOW-confidence and could not be verified here. A fresh literature audit
is mandatory before any paper claim. Absence from this audit is NOT proof of novelty.

---

## Engineering conclusion

The prior art does NOT block the AMPS0 **engineering experiment**. The mechanism (attribute-aware
BF16 optimizer-state storage fused into 3DGS) appears unoccupied. However — and decisively — the
**P0 headroom profile already shows insufficient optimizer headroom** (f_opt = 0.059-0.14, all
below the 0.20 gate). AMPS0 is therefore `NOVELTY_NOT_ESTABLISHED` as a *paper* claim (fresh audit
pending) and `DROP_INSUFFICIENT_OPTIMIZER_HEADROOM` as an *engineering* lane.
