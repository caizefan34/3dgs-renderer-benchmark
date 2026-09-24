# Phase 6.6 — Scope and Context Preservation

## Principle

Never allow `Experiment Result → universal FACT`. Every finding must carry scope, context, and uncertainty.

## Scope metadata fields

```
project_id          ☑ Always required
question_id         ☑ Optional
hypothesis_id       ☑ Optional
experiment_id       ☑ Optional
run_ids             ☑ Optional (list)
dataset             ☑ Optional
hardware            ☑ Optional
software            ☑ Dict (driver/cuda/pytorch)
configuration       ☑ Dict (tile_size, config_id, etc.)
code_commit         ☑ Optional
scope               ☑ Required (description of evaluated context)
created_at          ☑ Auto-generated
```

## Claim types

| ResearchFinding | KnowledgeClaim | Allowed? |
|----------------|---------------|----------|
| EXPERIMENT_RESULT | EXPERIMENT_RESULT | Yes |
| INFERENCE | INFERENCE | Yes |
| — | FACT | Only with explicit Knowledge Integrity Policy |
| — | CONTEXTUAL_OBSERVATION | Yes |

## Overgeneralization patterns detected

Patterns that trigger Integrity Gate WARNING:
- "is always"
- "is never"
- "always"
- "never"
- "universally" (when not preceded by "not")
- "under any condition"
- "in all cases"
- "is guaranteed"
- "definitively proves"

Properly qualified patterns that pass:
- "insufficient evidence that X is universally..."
- "under A100, tile16 is faster"
- "performance is workload-dependent"

## Real finding generated (from Phase 6.5 3DGS data)

```
"Under A100 with bicycle and garden scenes,
 there is insufficient evidence that tile32
 is universally faster than tile16."
```

This finding preserves:
- Hardware (A100)
- Dataset (mipnerf360)
- Scenes (bicycle, garden)
- Uncertainty (limited coverage)
- Contradicting evidence (tile16 leads on bicycle, tile32 on garden)
- Alternative explanations (scene complexity, coverage patterns)
