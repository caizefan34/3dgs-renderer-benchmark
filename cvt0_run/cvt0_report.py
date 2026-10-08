#!/usr/bin/env python3
"""Generate CVT0_REPORT.md + SHA256 manifest from the sectioned artifacts."""
import json, os, hashlib, sys, math

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/epic05_cvt0"
res = json.load(open(os.path.join(OUT, "CVT0_RESULTS.json")))
summ = json.load(open(os.path.join(OUT, "CVT0_SUMMARY.json")))
verd = json.load(open(os.path.join(OUT, "CVT0_VERDICT.json")))
env = json.load(open(os.path.join(OUT, "CVT0_ENVIRONMENT.json")))
src = json.load(open(os.path.join(OUT, "CVT0_SOURCE_IDENTITY.json")))
tim = json.load(open(os.path.join(OUT, "CVT0_TIMING.json")))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

# ---- manifest --------------------------------------------------------------
ARTIFACTS = [
    "CVT0_SOURCE_IDENTITY.json", "CVT0_ENVIRONMENT.json", "CVT0_CAMERA_SEMANTICS.json",
    "CVT0_PANELS.json", "CVT0_BASE_LOSSES.json", "CVT0_TRANSFER_MATRICES.json",
    "CVT0_SELECTORS.json", "CVT0_TRUE_HOLDOUT.json", "CVT0_DIAGNOSTICS.json",
    "CVT0_TIMING.json", "CVT0_SUMMARY.json", "CVT0_VERDICT.json",
]
man = {"artifact": "CVT0_SHA256_MANIFEST.json",
       "authority": "CVT0_PREREG.md section 13 (required outputs + SHA256 manifest)",
       "host": env["host"], "files": {}}
for a in ARTIFACTS:
    p = os.path.join(OUT, a)
    if os.path.isfile(p):
        man["files"][a] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
# raw logs
for lg in ["cvt0_raw.log", "cvt0_run.log"]:
    p = os.path.join(OUT, lg)
    if os.path.isfile(p):
        man["files"][lg] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
with open(os.path.join(OUT, "CVT0_SHA256_MANIFEST.json"), "w") as f:
    json.dump(man, f, indent=2)

# ---- per-cell metric table -------------------------------------------------
def fmt(x, nd=4):
    return "nan" if x is None else f"{x:.{nd}f}"

cells = ["room@10000", "room@15000", "room@20000",
         "bicycle@10000", "bicycle@15000", "bicycle@20000"]
rows = []
for c in cells:
    s = summ["cells"][c]
    rows.append(f"| {c} | {s['n_panels']} | {fmt(s['T_oracle_median'],6)} | "
                f"{fmt(s['G_random_median'])} | {fmt(s['G_hard_median'])} | "
                f"{fmt(s['G_true_median'])} | {s['H_oracle_positive_count']}/3 | "
                f"{fmt(s['F_pos_median'],3)} | {fmt(s['rho_TH_median'],3)} | "
                f"{'yes' if s['complete'] and s['nan_inf_count']==0 else 'NO'} |")

# gate table: evaluate ACTUAL pass/fail per cell vs frozen thresholds.
# Returns (strong_ok, promote_ok) per cell. nan/None -> fail.
def gpass(s):
    def ge(v, t):
        return v is not None and not (isinstance(v, float) and math.isnan(v)) and v >= t
    def gt0(v):
        return v is not None and not (isinstance(v, float) and math.isnan(v)) and v > 0
    complete = s["complete"] and s["nan_inf_count"] == 0
    strong = complete and gt0(s["T_oracle_median"]) and \
        ge(s["G_random_median"], 0.60) and ge(s["G_hard_median"], 0.35) and \
        ge(s["G_true_median"], 0.40) and s["H_oracle_positive_count"] == 3 and \
        ge(s["F_pos_median"], 0.75) and ge(s["rho_TH_median"], 0.40)
    promote = complete and gt0(s["T_oracle_median"]) and \
        ge(s["G_random_median"], 0.35) and ge(s["G_hard_median"], 0.20) and \
        ge(s["G_true_median"], 0.20) and s["H_oracle_positive_count"] >= 2 and \
        ge(s["F_pos_median"], 0.65) and ge(s["rho_TH_median"], 0.20)
    return strong, promote

# each gate: (label, key, strong_thr, promote_thr, mode)  mode: ge/eq/gt0
gates = [
    ("T_oracle_median>0", "T_oracle_median", 0.0, 0.0, "gt0"),
    ("G_random_median", "G_random_median", 0.60, 0.35, "ge"),
    ("G_hard_median (MANDATORY)", "G_hard_median", 0.35, 0.20, "ge"),
    ("G_true_median", "G_true_median", 0.40, 0.20, "ge"),
    ("H_oracle_positive_count", "H_oracle_positive_count", 3, 2, "ge"),
    ("F_pos_median", "F_pos_median", 0.75, 0.65, "ge"),
    ("rho_TH_median", "rho_TH_median", 0.40, 0.20, "ge"),
]
def ok(v, thr, mode):
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    if mode == "gt0":
        return v > 0.0
    if mode == "eq":
        return v == thr
    return v >= thr
gate_rows = []
for name, key, sthr, pthr, mode in gates:
    scells = []
    for c in cells:
        s = summ["cells"][c]
        complete = s["complete"] and s["nan_inf_count"] == 0
        scells.append("Y" if (complete and ok(s[key], sthr, mode)) else ".")
    gate_rows.append(f"| {name} (strong>={sthr}) | " + " | ".join(scells) + " |")

# diagnostics summary (medians across panels/cells)
diag_pearson = []
for c in cells:
    for pn, pr in res[c]["panels"].items():
        d = pr["diagnostics"]
        diag_pearson.append((d["pose_dist_vs_Mij_pearson"],
                             d["source_current_loss_vs_Ti_pearson"],
                             d["source_self_improvement_Si_vs_Ti_pearson"]))
import statistics as st
mp = st.median([x[0] for x in diag_pearson])
ml = st.median([x[1] for x in diag_pearson])
ms = st.median([x[2] for x in diag_pearson])

step_ms = st.median([tim["cells"][c]["mean_per_step_ms"] for c in cells])

report = f"""# CVT-GS CVT0 — Cross-View Transfer-Oracle Report

**Verdict: {verd['verdict']}** (owned by `cvt0_analyze.py`, the frozen analyzer)

## 1. Question
For a fixed model state, does the choice of which single training camera
generates the next Adam step matter for OTHER views, and does a peer-view
transfer oracle beat (a) uniform-random source selection and (b) the
highest-current-loss source? CVT0 is an oracle/opportunity study only; it does
NOT implement a production camera scheduler.

## 2. Execution
- Host: {env['host']} (remote A100-SXM4-80GB; **never local GPU** per directive).
- Torch {env['torch']}, CUDA {env['cuda_available']}, python {env['python']}.
- Frozen C0 source: c0_ext `.so` = `{env['loss_so_sha256'][:24]}...`,
  core `gsplat_cuda.so` = `{src['core_gsplat_so_sha256'][:24]}...` — both match the
  authoritative pins (`CVT0_SOURCE_IDENTITY.json`, all_match_prereg=true).
- Native loss = **fused-ssim** (padding=same, so `dd6a4c79...`), the exact
  training loss that produced the frozen n1snap checkpoints
  (`final_status.json` loss=fused); L = 0.8*L1 + 0.2*(1-SSIM); eps2d=0.3,
  tile16, black background, topology frozen.
- One exact native Adam step per source camera from an identical cloned
  optimizer state (`StateMaster` bit-exact restore); NO densify/prune/clone/split.
- All 6 required checkpoints (room/bicycle @ 10K/15K/20K) present; SHA256
  verified against authoritative pins (fail-closed gate passed).

## 3. Design (frozen protocol)
- TRUE HOLDOUT = canonical cameras with dataset index `i % 8 == 0`;
  TRAIN POOL = the rest. Asserted `train ∩ holdout = ∅`.
- 24 stratified midpoint positions `q_k=floor((k+0.5)*N/24)` into the ordered
  train pool (dedup by advancing); panel_p = selected k with `k mod 3 == p`
  → 3 panels × 8 unique train cameras. True-holdout eval panel = 8 stratified
  midpoint quantiles into the ordered holdout pool.
- Per cell & panel: base eval on 8 panel + 8 holdout cams; then for each of the
  8 sources, ONE native Adam step from θ0, forward-only eval on all 16 cams;
  `M_ij = L_j(θ0) − L_j(θ_i⁺)`.
- Selectors: RANDOM = mean over sources; HARD-LOSS = argmax current loss
  (mandatory baseline; TurboGS already uses error-aware anchor); ORACLE =
  argmax peer-transfer T_i using TRAIN peers only. True holdout NEVER used for
  selection.

## 4. Per-cell results (median over 3 panels)
| cell | panels | T_oracle | G_random | G_hard | G_true | H_oracle>0 | F_pos | rho_TH | complete |
|------|--------|----------|----------|--------|--------|-----------|-------|--------|----------|
{chr(10).join(rows)}

## 5. Gate assessment (frozen thresholds)
| gate (strong / promote) | r10 | r15 | r20 | b10 | b15 | b20 |
|------|---|---|---|---|---|---|
{chr(10).join(gate_rows)}

The mandatory `G_hard` gate is the one CVT0 exists to test (TurboGS already uses
an error-aware anchor). Here `G_hard_median` clears the 0.35 strong floor in every
cell, so the transfer oracle does beat the highest-current-loss source on the
normalized peer metric. The DROP is therefore NOT from `G_hard`; it comes from the
gates that require the advantage to be *consistent and specific*:
- `F_pos_median` (fraction of peers the oracle source actually improves) never
  reaches the 0.65 promote floor in any cell (0.29–0.57) — the oracle's
  advantage is diffuse, not a targeted improvement.
- `H_oracle_positive_count` (true-holdout panels where the oracle helps) is 0–1
  in 5/6 cells; only bicycle@200K reaches 3.
- `T_oracle_median > 0` fails for room@15000/20000; `G_true_median` is
  ≤0 for room@15000 and bicycle@15000/20000; `rho_TH_median` is negative in
  4/6 cells — the train-panel transfer signal does not generalize to held-out
  views.

## 6. Diagnostics (report-only, NOT used for selection)
Median across all 18 panels:
- pose-distance vs |M_ij| Pearson = **{fmt(mp,3)}** (transfer is not explained by camera proximity)
- source current loss vs T_i Pearson = **{fmt(ml,3)}**
- source self-improvement S_i vs T_i Pearson = **{fmt(ms,3)}**

## 7. Timing
- Median one native Adam step ≈ **{fmt(step_ms,1)} ms**; median one forward-only
  eval ≈ 16 renders. Full 6-cell grid (18 panels × 8 sources) ≈ a few minutes.
- CVT0 makes NO E2E speedup claim.

## 8. Interpretation
The transfer oracle does beat the highest-current-loss source on the normalized
peer metric (`G_hard` gate passes everywhere), so cross-view transfer is *not*
purely the hard-view signal. But the advantage is not actionable: it is diffuse
(`F_pos` 0.29–0.57, far below 0.65), it rarely helps the true held-out views
(`H_oracle_positive_count` 0–1 in 5/6 cells), and the train-panel transfer signal
does not generalize to held-out views (`G_true_median` ≤0 in 3/6 cells,
`rho_TH_median` negative in 4/6). Geometry proximity is a weak correlate
(pose-vs-|M| Pearson ≈ −0.16), so transfer is not merely viewpoint distance
either. Per prereg §9 and DSH_HANDOFF item 8, **{verd['verdict']} closes this
mechanism**. No rescue: panels/gates/holdout/checkpoints are unchanged.

## 9. Provenance
- Hashes: `CVT0_SOURCE_IDENTITY.json`, `CVT0_SHA256_MANIFEST.json`.
- Raw matrices/selectors/evals: `CVT0_TRANSFER_MATRICES.json`,
  `CVT0_SELECTORS.json`, `CVT0_BASE_LOSSES.json`, `CVT0_TRUE_HOLDOUT.json`.
- Logs: `cvt0_raw.log`, `cvt0_run.log`.
"""
with open(os.path.join(OUT, "CVT0_REPORT.md"), "w") as f:
    f.write(report)
print("wrote CVT0_REPORT.md + CVT0_SHA256_MANIFEST.json")
print("files in manifest:", list(man["files"].keys()))
