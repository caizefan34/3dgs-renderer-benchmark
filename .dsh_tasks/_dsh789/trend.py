import json
from pathlib import Path
RUNS = Path("/root/dsh_20261001/runs/DSH_PHASEA")
for s in ["room", "garden", "bicycle"]:
    print("===", s, "===")
    with (RUNS / s / "trace" / "phase_a_records.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            d8 = r["dsh8_apbc0"]
            frac = d8["unreachable_fraction_cost_weighted"]
            print("  iter=%6d unreach_frac=%.4f surv_buckets=%d total_visits=%d unreach_visits=%d max_contrib_p50=%s" % (
                r["iteration"], frac, d8["surviving_bucket_count"],
                d8["total_valid_pixel_visits"], d8["unreachable_pixel_visits"],
                r["dsh7_bars0"]["max_contrib"]["p50"]))
