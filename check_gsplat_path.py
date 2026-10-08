#!/usr/bin/env python3
import sys
print("Python path:")
for p in sys.path:
    print(f"  {p}")

import gsplat
print(f"\ngsplat location: {gsplat.__file__}")
print(f"gsplat version: {gsplat.__version__}")
