# GSC0 Preregistration

## Core hypothesis

Gaussian training states have heterogeneous temporal sensitivity.

## Not allowed

- global AMP claims
- tensor-wide FP16 rescue
- quality tuning after results
- changing training schedule
- changing densification rules

## Gate A: stability

At least 2/3 scenes must show a substantial stable Gaussian population under a fixed threshold chosen before final aggregation.

## Gate B: error

Candidate precision reduction must have bounded representation error.

## Gate C: economics

Estimated benefit must exceed complexity:

memory reduction or bandwidth reduction should be large enough to justify implementation.

If any gate fails:
DROP.

## Important distinction

GSC0 studies representation redundancy.

It is not:
- pruning
- Gaussian removal
- optimizer replacement
- loss approximation
