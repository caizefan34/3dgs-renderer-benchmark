# Single-birth sibling-coordinate diagnostic

Written before the GPU runs. This is not full training or novel-method evidence.

- Restore room/garden@2000 and saved Adam states explicitly. Freeze subsequent
  topology and SH degree. Same previous fixed 256 parents and three split draws
  (20260926..20260928), same cameras, original full-resolution photometric loss.
- Canonicalize every edited arm through the same FP32 encode/decode. Native
  controls store decoded physical values; coordinate arms retain encoded values.
  Require matching child SHA256 and initial losses for every matched seed.
- Before training each scene, require exact equality of all activated renderer
  inputs and the first A image; check actual renderer gradients against the
  shared/relative chain rule. Allow relative L2 < 1e-3 or absolute < 1e-8 due to
  GPU reduction variation. Also record old-parent/shared gradient alignment on
  that single camera; it does not establish alignment across training views.
- Seven policies per draw: native, pair_zero (basis only), pair_shared (parent
  moments on shared mode), pair_full (add fresh relative clock), physical
  inheritance .5/.25 and .2/.04, physical_lr2 (double newborn native updates).
  A native repeat per draw plus two no-edit arms totals 26 arms per scene.
- All pair modes use native group learning rates. The doubled physical update
  is a magnitude control, not an exact equivalence of Adam across coordinates.
- Evaluate steps 0,1,8,32,128,256. A contains original three candidate views; B
  contains 16 other views excluded from continuation, possibly seen before the
  checkpoint. Do not interpret B as official held-out generalization.
- Report each draw, paired native-repeat differences, A/B agreement, inheritance
  controls, and complete prototype step time including O(ND) decoding. Three
  draws in two development scenes cannot establish SOTA or statistical robustness.
- No repeated split/prune transport, compressed optimizer or CUDA rewrite until
  evidence supports the mechanism. No full-training speed claim from this probe.

Local independent checks: nonlinear physical-loss gradient chain rule and
fresh-relative updates against two separate PyTorch Adam optimizers: 2 passed.

## Recorded numerical amendment (after the first garden gate failed)

The original garden run stopped before any training arm: rotation/children
max absolute error 3.6294805e-6, relative L2 0.001674759, exceeding the fixed gate.
The original executed source remains `sibling_probe.py`; no result was discarded
from a training-policy comparison.

An independent four-native/four-encoded backward audit, with eight exactly
matching initial images, measured native rotation-child replay max absolute
2.2523745e-6 and L2 2.7116660e-6. Cross-basis maxima were 2.8094437e-6 and
4.0978398e-6; the difference of four-sample means had L2 4.6841288e-7.
This supports numerical reduction variation, not exact gradient identity or a
universal error bound. Audit source and evidence are preserved separately.

`sibling_probe_audited.py` uses the original fixed gate OR requires both absolute
and L2 error within 3 times independently measured native replay plus 1e-8. It
records which gate passed and hashes the numeric evidence. This explicit
amendment changes only validation/provenance, never an optimizer or training
budget. Room retains its original source and fixed-gate result. Garden training
starts only after this audit. Do not claim the amended gate was preregistered.
