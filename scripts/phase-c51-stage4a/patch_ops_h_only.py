#!/usr/bin/env python3
"""Patch only Ops.h — the declaration that ext.cpp uses."""
import sys
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/phase-c51-stage4a")
from patch_cuda import patch_ops_h
patch_ops_h()
print("Done patching Ops.h")
