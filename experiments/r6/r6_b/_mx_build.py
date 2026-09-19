#!/usr/bin/env python3
"""Build and smoke-test the R6-B patched gsplat on mx."""
import sys
print("Python:", sys.executable)
print("Building patched gsplat...")
from gsplat.cuda._backend import _C
print("BUILD OK")
print("has r6b_set_mode:", hasattr(_C, "r6b_set_mode"))
print("has r6b_last_scatter_ms:", hasattr(_C, "r6b_last_scatter_ms"))
print("has r6b_last_clear_ms:", hasattr(_C, "r6b_last_clear_ms"))
print("has r6b_metadata_bytes:", hasattr(_C, "r6b_metadata_bytes"))
print("has r6b_prev_n_rows:", hasattr(_C, "r6b_prev_n_rows"))
