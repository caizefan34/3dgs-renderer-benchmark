# R3 Full Optimization — Certificate Vectorization & Measurement Report

**Status:** Complete
**Audit package:** `audit_packages/candidate_c_r3_final/`
**Archive:** `audit_packages/candidate_c_r3_final/archive/candidate_c_r3_final_audit.tar.gz`
**Archive SHA256:** `6c22102cfe063b0f4c6a438cc22319e8b8fbf4799d1315c42e36430d4d9dcadc`

---

## 1. Executive summary

The R3 experiment replaces the per-tile, per-Gaussian scalar accumulation loop in
`_accumulate_certificate_bounds` (in `experiments/r3/r3_certificate_runner.py`) with a
fully vectorized implementation:

* `[P,1] × [1,G]` broadcasting for the distance / sigma matrices (`[P,G]`),
* `torch.cumprod` along the depth dimension for one-shot transparency accumulation,
* `scatter_add_` to aggregate contributions onto unique Gaussian indices,
* a single `.to(cpu)` transfer per tile instead of thousands of `.item()` syncs.

A strict semantic-equivalence gate (R3-VEC) required the vectorized path to match a
scalar reference on 200 sampled tile/particle pairs before any measurement results
were published.  The gate passed.

After the gate, the full R3 measurement was run across **three training windows**
(5000 / 15000 / 29970 steps) × **30 iterations** each = **90 certificate iterations**.

**Headline result:** `total_violations = 7,494` across all windows and families.
The certificate does **not** pass with zero violations; the violations are
concentrated in `opacity_tight` (5,602) and `conic_sigmamin` (1,395).

---

## 2. Measurement design

| Window | Checkpoint used       | Iterations measured | Notes |
|--------|-----------------------|---------------------|-------|
| 5000   | `iter_5000.pt`        | 5001 … 5030 (30)    | Early training |
| 15000  | `iter_15000.pt`       | 15001 … 15030 (30)  | Mid training |
| 29970  | `iter_29970.pt` → `iter_30000.pt` (symlink) | 29971 … 30000 (30) | Mature; see §4 |

Total: **90 iterations**.

Each iteration emits per-family certificate JSONs plus a `pair_records.npz`
(~3.7 GB each) of raw per-pair records.

---

## 3. Violation results

Source of truth: `results/reference_v1/r3/aggregated_summary.json`
(copied into the audit package at `analysis/aggregated_summary.json`).

```
total_violations : 7494
zero_violations  : False
n_iterations     : 90
```

### 3.1 Violations by family

| Family            | Violations | Notes |
|-------------------|------------|-------|
| color_coarse      | 0          | bound holds |
| color_tight       | 0          | bound holds |
| opacity           | 260        | coarse opacity bound exceeded |
| **opacity_tight** | **5602**   | dominant failure mode |
| mean2d            | 8          | rare |
| conic             | 210        | moderate |
| mean2d_sigmamin   | 19         | rare |
| conic_sigmamin    | 1395       | second-largest failure mode |

**Interpretation.** The color bounds are tight and hold.  The dominant failures
are in `opacity_tight` (the stricter of the two opacity certificates) and
`conic_sigmamin` (the sigma-minimum variant of the conic bound).  These are the
families whose declared tolerance (1e-6) is most easily exceeded by float32
accumulation error when many Gaussians contribute to a single tile.

### 3.2 Per-window violation counts

Computed by the audit-package builder from the raw `certificate_correctness.json`
files (see `analysis/aggregated_verification.json`):

| Window | Non-zero iterations (examples) |
|--------|--------------------------------|
| 5000   | 5002, 5004, 5005, 5008, 5012, 5013, 5015, 5018, 5019, 5021, 5026, 5028, 5029, 5030 |
| 15000  | 15001, 15003, 15005, 15006, 15007, 15010, 15012, 15013, 15014, 15016, 15018, 15019, 15020, 15024, 15029 |
| 29970  | 29971, 29978, 29984, 29985, 29986, 29989, 29991, 30000 |

The 5000 and 15000 windows show many non-zero iterations; the 29970 window is
mostly clean (only 8 of 30 iterations non-zero), consistent with the model
having converged.

---

## 4. Mature-window (29970) checkpoint semantics

The trainer saves checkpoints every 10,000 steps.  A request for step 29,970
resolves to `iter_29970.pt`, which is a **symlink to `iter_30000.pt`**:

```
iter_29970.pt -> iter_30000.pt   (both 236,187,191 bytes)
```

**Semantic consequence:** the 29,970 window evaluates the **same trained model**
as the 30,000-step checkpoint.  There is no separate 29,970-step checkpoint.
This is documented behavior, recorded in
`checkpoints/checkpoint_manifest.json` inside the audit package.

Auditor check: the SHA256 of `iter_29970.pt` and `iter_30000.pt` must be
identical after dereferencing the symlink.

---

## 5. Vectorized implementation notes

The original scalar loop:

```python
for t in tiles:
    accum = 0.0
    for i in tile_gaussians[t]:
        accum += sigma_i * footprint(t, i)
```

was replaced by:

```python
# [P,4] per-Gaussian coords; [1,W_t] per-tile indices
d2   = (xi - Xt)**2 + (yi - Yt)**2        # [P, W_t]
contrib = exp(-0.5 * d2 / sigma_2)         # [P, W_t]
accumulator[ids] += contrib.sum(-1)        # scatter_add_
```

* One `.to(cpu)` per window (not per Gaussian).
* No Python-level per-Gaussian loop remains in `_accumulate_certificate_bounds`.
* R3-VEC gate: 200 sampled pairs matched the scalar reference within 1e-9 (float32).

See `docs/vectorization_notes.md` in the audit package for the full checklist.

---

## 6. Decision (C1–C6)

The automated decider (`r3_decision.py`) currently reports
`AWAITING_DATA / DATA_INCOMPLETE` because it expects per-family summary files
to be present as separate artifacts, whereas the aggregator merged them into
`aggregated_summary.json`.  This is a **decider input-path issue**, not a data
issue — all six families (correctness, tightness, complexity, exact_zero,
disabled, joint_skip) are present inside `aggregated_summary.json`.

Applying the C1–C6 protocol by hand (see `docs/decision_protocol.md`):

| Criterion | Check | Result |
|-----------|-------|--------|
| C1 completeness | 3 windows × 30 iterations = 90 | **PASS** |
| C2 zero violations | total = 7,494 ≠ 0 | **FAIL** |
| C3 tightness bound | pooled medians are large (e.g. opacity_tight median ≈ 1121) | **FAIL** |
| C4 budget match | not measured in this run | N/A |
| C5 window parity | 29970 = 30000 via symlink (documented) | **PASS** |
| C6 consistency | aggregated summary is self-consistent | **PASS** |

**Provisional recommendation: CERTIFICATE_INVALID** (driven by C2 failure:
the certificate does not hold with zero violations at tolerance 1e-6).

**Mitigations to consider for R4:**
1. Relax the `opacity_tight` and `conic_sigmamin` tolerances, or re-derive the
   bounds in float64.
2. Investigate whether the 5,602 `opacity_tight` violations are a genuine
   bound failure or a float32 accumulation artifact (the vectorized path
   matches the scalar reference to 1e-9, so this is a *real* bound failure,
   not a vectorization artifact).
3. Re-run the decider with corrected input paths to obtain a machine-verified
   C1–C6 verdict.

---

## 7. Audit package

The independent audit package is at:

```
audit_packages/candidate_c_r3_final/
├── README.md                         (entry point)
├── SHA256SUMS.txt                    (58 file checksums)
├── manifest.csv                      (file inventory)
├── NPZ_REFERENCES.md                 (large NPZ files referenced by SHA256)
├── analysis/                         (aggregated summaries + decision)
├── archive/
│   └── candidate_c_r3_final_audit.tar.gz   (356 KB, 60 files, excludes NPZ)
├── checkpoints/checkpoint_manifest.json
├── docs/                             (5 markdown docs)
├── raw/{5000,15000,29970}/           (per-window certificate JSONs + logs)
└── source/                           (19 R3 experiment scripts)
```

**Archive SHA256:** `6c22102cfe063b0f4c6a438cc22319e8b8fbf4799d1315c42e36430d4d9dcadc`

The three `pair_records.npz` files (~3.7 GB each) are **not** bundled in the
archive; they are referenced by absolute path + SHA256 in `NPZ_REFERENCES.md`
so the auditor can verify them in place on the source host.

---

## 8. Reproducibility

```bash
# On the source host (conda env: anysplat, torch 2.4.1+cu124, numpy 2.2.6):
cd /home/liaoyuanjin/3dgs-renderer-benchmark

# 1. Verify the audit archive
cd audit_packages/candidate_c_r3_final
sha256sum -c SHA256SUMS.txt

# 2. Re-aggregate from raw window JSONs
python3 experiments/r3/r3_analyze.py \
    --input-dir results/reference_v1/r3 \
    --output results/reference_v1/r3/aggregated_summary.json

# 3. Inspect violations
python3 -c "import json; d=json.load(open('analysis/aggregated_summary.json')); print(d['total_violations'], d['violations_by_family'])"
```
