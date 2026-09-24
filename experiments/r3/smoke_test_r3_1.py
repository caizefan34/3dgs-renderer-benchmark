#!/usr/bin/env python3
"""Smoke test: import repaired runner and check function signature."""
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("runner", "experiments/r3/r3_certificate_runner.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("IMPORT_OK")
sig = mod._accumulate_tile_bounds.__code__.co_varnames[:6]
print("_accumulate_tile_bounds first args:", sig)
print("compute_C_max_t args:", mod.compute_C_max_t.__code__.co_varnames[:5])
