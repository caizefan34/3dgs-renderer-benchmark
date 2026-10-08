"""Run the entire LAVOS0 offline CPU test suite.

Usage:  python tests/run_all.py
Exit code 0 iff every test passes. This is the deterministic gate that the
harness logic is correct independent of any A100 run.
"""

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

TEST_FILES = [
    "test_adam_parity.py",
    "test_topology_oracle.py",
    "test_debt_replay.py",
    "test_perf_oracle.py",
    "test_event_order_and_verdict.py",
    "test_pipeline_integration.py",
]


def _load_module(path):
    name = os.path.splitext(os.path.basename(path))[0]
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    total = 0
    failed = 0
    for tf in TEST_FILES:
        path = os.path.join(HERE, tf)
        mod = _load_module(path)
        fns = [getattr(mod, k) for k in dir(mod)
               if k.startswith("test_") and callable(getattr(mod, k))]
        for fn in sorted(fns, key=lambda f: f.__name__):
            total += 1
            try:
                fn()
                print(f"PASS {tf}::{fn.__name__}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL {tf}::{fn.__name__}")
                traceback.print_exc()
    print(f"\n{total - failed}/{total} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
