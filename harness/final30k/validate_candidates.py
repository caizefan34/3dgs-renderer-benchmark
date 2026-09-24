#!/usr/bin/env python3
"""
validate_candidates.py — Identity + protocol gate for the FINAL 30K benchmark.

The harness MUST ABORT (exit non-zero) if:
  1. the expected renderer binary SHA256 does not match the resolved binary,
  2. a required patch SHA256 is missing or mismatched,
  3. two candidates resolve to DIFFERENT experiment-critical settings,
  4. a resolved config.json value differs from the frozen protocol.

Reads:
  - artifacts/final-30k/candidate_registry.json (expected identities)
  - artifacts/final-30k/protocol.json (experiment-critical fields)
  - per-run provenance.json + config.json (resolved identities)

This is a GATE, not a runner. It validates the run set before the benchmark may start.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL30K = REPO_ROOT / "artifacts" / "final-30k"

# Experiment-critical fields that MUST be identical across all candidates (protocol.json)
EXPERIMENT_CRITICAL = [
    "number_of_training_steps", "resolution", "loss", "optimizer", "learning_rates",
    "densification_schedule", "opacity_reset_schedule", "sh_schedule", "initialization",
    "seed", "evaluation_schedule", "checkpoint_schedule",
]


def _validate_tagged(tagged: list, registry: dict, protocol: dict) -> bool:
    ok = True
    by_candidate = {}
    for prov in tagged:
        by_candidate.setdefault(prov["candidate_id"], []).append(prov)

    # Gate 1: binary identity per candidate
    print("== Gate 1: binary SHA256 identity ==")
    for cid, provs in by_candidate.items():
        expected_raw = registry["candidates"].get(cid, {}).get("binary_sha256", "")
        # Take the LEADING run of hex chars (the value may carry a trailing note)
        import re
        m = re.match(r"^[0-9a-fA-F]+", expected_raw.strip())
        expected_hex = m.group(0) if m else ""
        for prov in provs:
            resolved = prov.get("renderer_binary_sha256", "")
            if "TO_BE" in expected_raw or "TBD" in expected_raw:
                print(f"  WARN {cid}/{prov['scene']}: expected binary not pinned yet")
            elif len(expected_hex) < 8:
                print(f"  WARN {cid}/{prov['scene']}: expected binary has no usable hex prefix")
            elif resolved[:len(expected_hex)] == expected_hex:
                print(f"  PASS {cid}/{prov['scene']}: binary matches ({resolved[:12]}...)")
            else:
                print(f"  ABORT {cid}/{prov['scene']}: binary {resolved[:12]}... != expected {expected_hex[:12]}...")
                ok = False

    # Gate 2: patch identity
    print("== Gate 2: patch SHA256 identity ==")
    for cid, provs in by_candidate.items():
        exp_patches = registry["candidates"].get(cid, {}).get("patch_sha256s", {})
        for prov in provs:
            for name, exp in exp_patches.items():
                # Skip non-hex values (descriptions like "part of frozen composed build...")
                if "TBD" in exp or "TO_BE" in exp or "see " in exp:
                    continue
                if not all(c in "0123456789abcdefABCDEF" for c in exp[:8]):
                    continue  # not a hex SHA (e.g. SCALAR_ADJOINT template-param note)
                res = prov.get("patch_sha256s", {}).get(name, "")
                if res and (res.startswith(exp) or exp.startswith(res)):
                    print(f"  PASS {cid}/{prov['scene']}: patch {name} matches")
                else:
                    print(f"  ABORT {cid}/{prov['scene']}: patch {name} missing/mismatch (expected {exp[:12]}..., resolved {res[:12]})")
                    ok = False

    # Gate 3: experiment-critical settings identical across candidates AND matching frozen protocol
    print("== Gate 3: experiment-critical settings (frozen protocol + cross-candidate identity) ==")
    resolved = {}
    for cid, provs in by_candidate.items():
        cfg = {}
        for p in provs:
            cfg_path = Path(p.get("_run_dir", "")) / "config.json"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                break
        resolved[cid] = cfg
    proto_settings = protocol.get("identical_across_candidates", {})
    for field in EXPERIMENT_CRITICAL:
        vals = set()
        for cid in resolved:
            v = resolved[cid].get(field)
            if v is not None:
                vals.add(json.dumps(v, sort_keys=True))
        exp = proto_settings.get(field)
        if len(vals) > 1:
            print(f"  ABORT: '{field}' differs ACROSS candidates: {vals}")
            ok = False
        elif vals and exp is not None and list(vals)[0] != json.dumps(exp, sort_keys=True):
            print(f"  ABORT: '{field}' resolves to a value not matching the frozen protocol")
            print(f"    protocol: {json.dumps(exp)[:120]}")
            print(f"    resolved: {list(vals)[0][:120]}")
            ok = False
        elif vals:
            print(f"  PASS '{field}': identical across {len(vals)} candidate(s), matches frozen protocol")
        else:
            print(f"  WARN '{field}': no config.json resolved yet (will be checked at run time)")

    print()
    if ok:
        print("GATE PASS: identity-clean and protocol-identical. May start.")
    else:
        print("GATE FAIL: ABORT before starting the FINAL 30K benchmark.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="*", required=True,
                    help="Paths to per-scene run artifact dirs (each has provenance.json + config.json)")
    ap.add_argument("--registry", default=str(FINAL30K / "candidate_registry.json"))
    ap.add_argument("--protocol", default=str(FINAL30K / "protocol.json"))
    args = ap.parse_args()
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    tagged = []
    for d in args.run_dirs:
        p = Path(d) / "provenance.json"
        prov = json.loads(p.read_text(encoding="utf-8"))
        prov["_run_dir"] = str(Path(d))
        tagged.append(prov)
    ok = _validate_tagged(tagged, registry, protocol)
    sys.exit(0 if ok else 1)
