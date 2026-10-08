"""Summarize prefixed JSON records from a FastGS live selection audit log."""

import argparse
import json
import re
from pathlib import Path


def summarize(records):
    actual = sum(record["actual_total"] for record in records)
    projected = sum(record["actual_projected_ge_3"] for record in records)
    concentrated = sum(
        record["actual_projected_ge_3_effective_views_lt_1_5"] for record in records
    )
    top_view = sum(record["actual_top_view_share_ge_0_75"] for record in records)
    return {
        "events": len(records),
        "actual_selection_events": actual,
        "projected_ge_3_selection_events": projected,
        "projected_ge_3_concentrated_events": concentrated,
        "concentrated_fraction_of_projected_ge_3": concentrated / projected if projected else None,
        "top_view_share_ge_0_75_events": top_view,
        "top_view_fraction_of_actual": top_view / actual if actual else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    source = args.log.read_text(encoding="utf-8", errors="replace")
    records = [
        json.loads(match)
        for match in re.findall(r"FASTGS_LIVE_SELECTION (\{[^}]+\})", source)
    ]
    if [record["event_index"] for record in records] != list(range(1, len(records) + 1)):
        raise AssertionError("missing or duplicate event indices")
    report = {
        "source_log": str(args.log),
        "all": summarize(records),
        "events_1_15": summarize(records[:15]),
        "events_16_30": summarize(records[15:30]),
        "events_31_45": summarize(records[30:45]),
        "first_event": records[0] if records else None,
        "last_event": records[-1] if records else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
