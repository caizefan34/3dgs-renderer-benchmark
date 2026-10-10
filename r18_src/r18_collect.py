#!/usr/bin/env python3
"""R18 collect — summarize step exit codes + key verdicts into RUN_STATUS.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r18_common import OUT, write_json


def _load(path):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def main():
    out = OUT
    codes_path = out / "logs" / "step_exit_codes.txt"
    steps = {}
    if codes_path.is_file():
        for line in codes_path.read_text().splitlines():
            if " " in line:
                k, v = line.rsplit(" ", 1)
                try:
                    steps[k] = int(v)
                except ValueError:
                    pass
    required = ["preflight", "p0_integrity", "p1_correctness", "p2_performance",
                "p3_verdict", "collect"]
    executed = [s for s in required if s in steps]
    not_executed = [s for s in required if s not in steps]
    all_zero = all(steps.get(s) == 0 for s in required) and not not_executed

    p0 = _load(out / "R18_P0_INTEGRITY_AUDIT.json")
    p1 = _load(out / "raw" / "R18_P1_details.json")
    p2 = _load(out / "raw" / "R18_P2_details.json")
    verdict = _load(out / "R18_RESEARCH_VERDICT.json")

    summary = {
        "summary": {"required": len(required), "executed": len(executed),
                    "not_executed": not_executed,
                    "all_steps_exit_zero": all_zero},
        "step_exit_codes": steps,
        "r18_out": str(out),
        "p0_binaries_all_match": (p0 or {}).get("binaries_all_match"),
        "p0_checkpoint_match": (p0 or {}).get("checkpoint", {}).get("match"),
        "p1_config_status": {r.get("config"): r.get("status")
                             for r in (p1 or {}).get("rows", [])},
        "p2_config_backend_status": {f"{r.get('config')}/{r.get('backend')}": r.get("status")
                                     for r in (p2 or {}).get("rows", [])},
        "verdicts": (verdict or {}).get("verdicts"),
        "missing_inputs": (verdict or {}).get("missing_inputs"),
    }
    write_json(out / "RUN_STATUS.json", summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
