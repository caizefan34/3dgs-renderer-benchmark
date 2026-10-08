P13-Y return package (remote host EPIC-05, /root/epic05_p13y)
Date 2026-09-27. Decision class: DROP_PERSISTENT_LINEARIZATION.

CONTENTS
  P13Y_package.tar.gz            artifacts/*.json (all gates/verdicts/plans),
                                 P13Y_REPORT.md, logs/ (all unit logs incl.
                                 P13Y_CONTAMINATION.log), the exact deployed
                                 scripts (*.py) and scripts/*.sh
  P13Y_pair_results_json.tar.gz  pair_results/{room,bicycle,garden}/pair_*.json
                                 + _SUMMARY.json, plus
                                 pair_results/_invalid_run1_harness_bug/
                                 (run-1 harness-bug outputs, kept verbatim)
  P13Y_run_metadata.tar.gz       3 x 17K capture runs + 5 x 400-iter OFF
                                 ensemble runs + the ON short run: results.json,
                                 timing/quality/training_results/curve,
                                 cam_log.json, p13y_iter_log.json,
                                 p13y_capture_identity.json,
                                 states/iXXXXX/meta.json (all 47),
                                 probes/probe_iXXXXX/probe_meta.json

LANDMARK ARTIFACTS (also inside P13Y_package.tar.gz)
  artifacts/P13Y_VERDICT.json                 decision class + next step
  artifacts/P13Y_Y23_AGG.json                 per-scene gates + pair table
  artifacts/P13Y_ADAPTER_PARITY_{scene}.json  Sec.5 gate: PASS, all_bitwise
  artifacts/P13Y_CAPTURE_NEUTRALITY.json      Sec.4 frozen criterion: FAIL
  artifacts/P13Y_CAPTURE_NEUTRALITY_trio3500.json, _SHORT.json  (as recorded)
  artifacts/P13Y_NEUTRALITY_DIAG_room.json    diagnosis of the FAIL
  artifacts/P13Y_NEUTRALITY_ENSEMBLE.json     5-OFF vs 1-ON diagnostic
  artifacts/P13Y_STATE_VERIFY.json            47 states re-verified, 0 bad
  artifacts/P13Y_SEQUENCE_INDEXING.json       L1/L2/L3 indexing proof
  artifacts/P13Y_SOURCE_IDENTITY.json         Sec.1 identity, all match
  artifacts/MANIFEST_P13Y.json                556 files / 30.12 GB on host

DELIBERATELY NOT SHIPPED (large; sha256 for every one of them is in
MANIFEST_P13Y.json and can be re-fetched from the host on demand)
  runs/*/states/iXXXXX/state.pt (47 x 0.2-0.7 GB raw optimizer tensors)
  runs/*/probes/probe_iXXXXX/trainer_frame.pt (trainer frames)
  pair_results/*/y1_snapshots/pair_*/**.npz (raw Y1 meta snapshots)

VERIFY AFTER UNPACK
  sha256sum -c TARBALL_SHA256.txt
  python -c "import json;print(json.load(open('artifacts/P13Y_VERDICT.json'))['class'])"
  -> DROP_PERSISTENT_LINEARIZATION
