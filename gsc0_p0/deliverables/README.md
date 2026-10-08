# GSC0 P0 — Gaussian State Compression: Redundancy Oracle

Status: **P0 ORACLE COMPLETE**. Verdict: **DROP_NO_STABLE_STATE**.

Local Windows machine = control-plane only. All training / rendering / CUDA / JIT /
trace collection executed on **EPIC-05** (`dsw-933622-64cc845c7d-lqswc`, 8x A100-SXM4-80GB).

## Deliverables (this folder)

| file | contents |
|---|---|
| `GSC0_VERDICT.json` | Preregistered gates A/B/C + final verdict |
| `GSC0_SUMMARY.json` | Per-scene stability / gradient / visibility / gate detail |
| `GSC0_REPORT.md` | Human-readable report (all distributions, nuance, provenance) |
| `GSC0_TRACE_INDEX.json` | Raw trace index: per-phase snapshots + sha256 + provenance |
| `GSC0_ERROR_ORACLE.json` | FP16/BF16 representation + optimizer-state error oracles |
| `MANIFEST.sha256` | sha256 of the 5 deliverables + 3 scripts + 9 raw snapshots |
| `scripts/` | `gsc0_trace.py` (harness), `gsc0_analysis.py`, `gsc0_report.py`, `gsc0_make_manifest.py` |

## Raw artifacts (on EPIC-05, not pulled locally)

Per-scene full-state snapshots live on EPIC-05 at `/root/gsc0_run/<scene>/traces/<scene>_<phase>_iter*.pt` (9 files, ~10 GB). Their sha256 is recorded in both `GSC0_TRACE_INDEX.json` and `MANIFEST.sha256` and was re-verified on EPIC-05 (9/9 match).

## Verification

Local (control dir):
```
# from this folder
sha256sum -c MANIFEST.sha256   # verifies the 5 deliverables + 3 scripts (8 entries);
                               # the 9 raw/ entries are on EPIC-05
```
EPIC-05 (raw snapshots): re-hash each `.pt` and compare to `GSC0_TRACE_INDEX.json` -> 9/9 match.

## Headline finding

Over the preregistered mature window (iter 15000 -> 30000, frozen topology), whole-Gaussian
state does NOT stabilize: only ~0.1% of Gaussians hold all five attributes (xyz, scale,
rotation, SH, opacity) within TAU=1e-2. Stability is strongly attribute-heterogeneous:
**position (xyz) is ~99%+ stable**, but scale/rotation/SH/opacity keep drifting 5-30%+.
Hence Gate A (stability) fails -> DROP_NO_STABLE_STATE. Gate B (the tiny stable subset IS
representable within bounds) and Gate C (negligible memory benefit) also fail. The
actionable signal for follow-up is position-only redundancy (a GSC1 per-attribute question).

## Preregistration

Thresholds fixed BEFORE aggregation (see `GSC0_VERDICT.json:prereg`): TAU=1e-2,
STABLE_FRAC_MIN=0.50, SCENE_FRAC_MIN=2/3, ECON_FRAC_MIN=0.40, ECON_COUNT_MIN=10000.
No threshold search, no quality tuning, no reduced-precision training run.