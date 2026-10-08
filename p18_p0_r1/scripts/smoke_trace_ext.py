#!/usr/bin/env python3
"""P18-P0-R1 smoke test: load trace-instrumented extension, call p18_trace_set."""
import importlib.util
import sys

import torch  # noqa: E402  (must precede .so dlopen for libc10/libtorch symbols)

SO = "/root/epic05_p18_r1/build_trace/experimental_gaussian_render_inference_scene_cuda.so"

spec = importlib.util.spec_from_file_location(
    "experimental_gaussian_render_inference_scene_cuda", SO)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.modules["experimental_gaussian_render_inference_scene_cuda"] = mod

print("[smoke] module loaded:", mod)
assert hasattr(mod, "p18_trace_set"), "p18_trace_set binding missing"
print("[smoke] has p18_trace_set: True")

# null pointers -> p18_on == False inside the kernel
mod.p18_trace_set(0, 0, 0, 0, 255)
print("[smoke] p18_trace_set(0,0,0,0,255) OK (trace off state)")

print("[smoke] torch", torch.__version__, "cuda avail:", torch.cuda.is_available())
try:
    print("[smoke] torch.ops ns:", [n for n in dir(torch.ops)
                                    if "gaussian" in n.lower() or "experimental" in n.lower()])
except Exception as e:
    print("[smoke] torch.ops listing failed:", e)
print("SMOKE_OK")
