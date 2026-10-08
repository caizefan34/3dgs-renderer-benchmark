import sys
sys.path.insert(0, "src")
try:
    import gsplat
    print(f"gsplat: {gsplat.__file__}")
except Exception as e:
    print(f"gsplat import failed: {e}")

try:
    import higs
    print(f"higs: {higs.__file__}")
except Exception as e:
    print(f"higs import failed: {e}")
