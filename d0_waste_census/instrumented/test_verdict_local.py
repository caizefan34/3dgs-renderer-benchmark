"""Synthetic test for finalize_verdict.py against the canonical tool."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "scripts"))


def run_case(name, ci_low, rows, expect):
    tmp = tempfile.mkdtemp(prefix="d0_verdict_")
    tv = {"artifact": "D0_TRACE_VALIDITY_30k.json", "verdict": "D0_TRACE_VALID_30K",
          "reasons": {"room": [], "bicycle": []}}
    cost = {"runs": {s: {"accounting": {"reconstruction_error": 0.001,
                                        "reconstruction_gate_ok": True,
                                        "sumcheck_gate_ok": True}}
                     for s in ("room", "bicycle")}}
    head = {"runs": {s: {"R_upper": 0.8, "R_upper_ci95": [ci_low, 0.9],
                         "S_upper_at_ci95_low": 4.0, "t_star": 25000,
                         "r_I_dead": 0.5, "r_N_dead": 0.5}
                     for s in ("room", "bicycle")}}
    for fn, obj in (("D0_TRACE_VALIDITY_30k.json", tv),
                    ("D0_COST_ACCOUNTING.json", cost),
                    ("D0_HEADROOM.json", head)):
        with open(os.path.join(tmp, fn), "w") as f:
            json.dump(obj, f)
    if rows is not None:
        with open(os.path.join(tmp, "D0_PREDICTABILITY_ROWS.json"), "w") as f:
            json.dump(rows, f)
    rc = subprocess.call([sys.executable,
                          os.path.join(SCRIPTS, "finalize_verdict.py"),
                          "--art-dir", tmp])
    assert rc == 0, rc
    with open(os.path.join(tmp, "D0_VERDICT.json")) as f:
        got = json.load(f)["verdict"]
    assert got == expect, (name, got, expect)
    print("OK", name, got)
    shutil.rmtree(tmp)


strong = [{"model": "logreg", "train_scene": "room", "test_scene": "bicycle",
           "auroc": 0.9, "test_survivor_fpr": 0.01,
           "test_dead_cost_weighted_recall": 0.7},
          {"model": "logreg", "train_scene": "bicycle", "test_scene": "room",
           "auroc": 0.88, "test_survivor_fpr": 0.02,
           "test_dead_cost_weighted_recall": 0.6}]
weak = [dict(strong[0], auroc=0.7), strong[1]]

run_case("strong_both", 0.75, strong, "D0_HEADROOM_AND_PREDICTABLE")
run_case("headroom_not_predictable", 0.75, weak,
         "D0_HEADROOM_BUT_NOT_PREDICTABLE")
run_case("no_headroom", 0.55, strong, "D0_NO_3X_HEADROOM")
run_case("borderline", 0.65, strong, "D0_INCONCLUSIVE")
run_case("no_pred_file", 0.75, None, "D0_HEADROOM_BUT_NOT_PREDICTABLE")
print("FINALIZE_VERDICT_SELFTEST_OK")
