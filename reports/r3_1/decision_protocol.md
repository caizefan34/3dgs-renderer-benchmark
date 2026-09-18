# Decision Protocol — CR1–CR4 (from r3_decision.py source code)

**Source of truth:** `experiments/r3/r3_decision.py` (frozen commit `32ab80e`)

---

## CR1 — Correctness

**Check:** Zero aggregate certificate violations across ALL families and ALL
iterations.

**Families checked (8):**
- Geometry primary: `mean2d`, `conic`, `mean2d_sigmamin`, `conic_sigmamin`
- Appearance secondary: `color_coarse`, `color_tight`, `opacity`, `opacity_tight`

**Data source:** `certificate_correctness.json` → `correctness[iter][family].violation_count`

**Pass condition:** `violation_count == 0` for every available family in every iteration.

---

## CR2 — JOINT Weighted-Work Removal

**Check:** JOINT weighted derivative-work removal fraction at 5% budget.

**Data source:** `tile_gaussian_certificate.json` → `joint_skip[iter][eps_key]`

**Metrics extracted:**
- `JOINT_SKIP_WEIGHTED_WORK_FRACTION` — primary gate
- `JOINT_SKIP_PAIR_FRACTION` — secondary
- `LOSS_CONDITIONED_NONZERO_SUPPORT_CULLING` — Candidate C novelty signal
  (excludes exact-zero prior-art support culling)

**Thresholds:**
- `>= 30%` → CR2 PASS (C_KEEP eligible)
- `10% – 30%` → CR2 MODERATE (C_MODIFY eligible)
- `< 10%` → CR2 FAIL (C_DROP)

**Budget levels reported:** 0.5%, 1%, 2%, 5%

---

## CR3 — Signal Coverage

**Check:** Signal persists across at least 3 canonical windows.

**Data source:** Union of iteration keys from `correctness` and `joint_skip`.

**Pass condition:** `len(windows_seen) >= 3`

---

## CR4 — SPD Certifiability

**Check:** SPD (Symmetric Positive Definite) disabled fraction is small enough
that the certificate covers the vast majority of Gaussians.

**Data source:** `certificate_disabled.json` → `disabled[iter].spd_disabled_count`

**Pass condition:** `disabled_fraction < 10%`

---

## Final Decision Logic

```
if CR1 pass AND CR2 pass (>=30%) AND CR3 pass AND CR4 pass:
    → C_KEEP
elif CR1 pass AND CR2 moderate (10-30%) AND CR3 pass:
    → C_MODIFY
else:
    → C_DROP
```

**Key rule:** CR1 failure (any violation) → automatic C_DROP, regardless of
opportunity numbers. Correctness is a hard gate.

---

## Critical Distinction (Red-Team Req 4)

- **EXACT_ZERO_SUPPORT_CULLING** = alpha < 1/255 pairs
  → Prior art (Speedy-Splat, AccuTile). NOT counted as Candidate C novelty.
- **LOSS_CONDITIONED_NONZERO_SUPPORT_CULLING** = pairs inside nonzero support
  that are skippable due to loss-conditioned derivative bounds.
  → THIS is Candidate C's primary novelty signal.

---

## Input Requirements

`r3_decision.py --input <DIR>` expects `<DIR>` to contain these 6 files:
1. `certificate_correctness.json`
2. `certificate_tightness.json`
3. `complexity_accounting.json`
4. `exact_zero_statistics.json`
5. `certificate_disabled.json`
6. `tile_gaussian_certificate.json`

The script writes `final_decision.json` into `<DIR>`.
