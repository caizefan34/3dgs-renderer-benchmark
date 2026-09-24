"""Check resize function signature."""
import inspect, sys, os
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo, "src"))
from benchmark_framework import resize_cameras
sig = inspect.signature(resize_cameras)
print(f"Signature: {sig}")
src = inspect.getsource(resize_cameras)
print(src[:2000])
