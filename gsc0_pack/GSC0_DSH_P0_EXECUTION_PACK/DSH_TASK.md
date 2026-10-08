# GSC0 DSH P0 Task

## Order

### GSC-0 Remote guard
Verify:
- hostname
- GPU
- cwd
- remote identity

If not remote:
STOP_REMOTE_GUARD.

### GSC-1 Source audit

Record:
- Gaussian parameter layout
- optimizer state layout
- densification/pruning behavior
- parameter replacement events
- precision assumptions

No code modification.

### GSC-2 Trace collection

Collect from EPIC-05 only.

Scenes:
- room
- garden
- bicycle

Phases:
- early training
- around densification completion
- mature checkpoint

For every Gaussian collect:

Geometry:
- xyz delta
- scale delta
- rotation delta

Appearance:
- SH delta

Other:
- opacity delta
- visibility frequency
- gradient magnitude history

### GSC-3 Stability analysis

Compute:

relative change:

    ||x_t - x_(t-k)|| / max(||x_t||, eps)

gradient stability:

    ||g_t-g_(t-k)||

visibility stability:

    active_count history

Report distributions:
- median
- p90
- p99
- scene comparison

### GSC-4 Precision error oracle

Simulate:

- FP32 reference
- FP16/BF16 candidate
- attribute-specific conversion

Estimate:
- representation error
- accumulated drift
- optimizer-state error

No training run with reduced precision in P0.

### GSC-5 Gate

Continue only if:

1. Significant fraction of Gaussian states satisfy a fixed error bound.
2. Stability appears in >=2/3 scenes.
3. Estimated memory/bandwidth benefit is meaningful.

Otherwise DROP.

No threshold search after observing results.

## Required outputs

GSC0_VERDICT.json
GSC0_SUMMARY.json
GSC0_REPORT.md
GSC0_TRACE_INDEX.json
GSC0_ERROR_ORACLE.json
MANIFEST.sha256
