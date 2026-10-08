# DSH task: execute EPIC-05 publication closure

You are executing a frozen publication validation campaign on EPIC-05 (8× A100-SXM4-80GB).
Do not invent new optimization branches and do not change C0 semantics.

Your job:
1. Inspect `/root/3dgs-renderer-benchmark` and the EPIC runtime.
2. Reconstruct the frozen publication trainer and B1A/C0 binaries from the repository evidence.
3. Freeze source/build provenance before timing.
4. Run P0 GPU calibration.
5. Run all groups in `epic05_manifest.json` with the same-GPU group policy.
6. Preserve every failed/contaminated attempt; rerun clean rather than deleting it.
7. Run P5 H8 off/on correctness+timing on room/bicycle/garden.
8. Run P6 H8 generality validation in a pristine gsplat or isolated drop-in backward path.
9. Produce machine-readable aggregates and a final scientific report.

Hard rules:
- no result may be called publication-grade without clean GPU snapshots;
- never mix EPIC numbers into the original PCIe aggregate;
- no “cross-architecture” wording: both systems are A100/SM80;
- no quality-improvement claim from drjohnson;
- no attribution of the full C0/B1A gap to F9/Scalar/H8;
- no new H9/H10 optimization work;
- any source or semantic mismatch stops the campaign before long runs.

Expected final verdict files:
- `EPIC05_P0_CALIBRATION.json`
- `EPIC05_P1_MAIN.json`
- `EPIC05_P2_ABLATION.json`
- `EPIC05_P3_EPS2D.json`
- `EPIC05_P4_MULTISEED.json`
- `EPIC05_P5_H8_MECHANISM.json`
- `EPIC05_P6_H8_GENERALITY.json`
- `EPIC05_FINAL_VERDICT.json`
