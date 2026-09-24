#!/usr/bin/env python3
"""RUNS IN A SUBPROCESS. Times ONE probe .so on the room/bicycle/garden fixtures
using the same protocol as p3_h_run timing. Writes {scene: [per-rep ms]}."""
import json, os, sys
from pathlib import Path
import importlib.util
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun")
import p3_h_run as H
import torch

so    = os.environ["PROBE_SO"]
modname = "experimental_p3h_prod_cuda" if "prod" in os.environ["PROBE_KEY"] else "experimental_p3h_v0_cuda"
scenes = os.environ["SCENES"].split(",")
reps  = int(os.environ["REPS"]); samples = int(os.environ["SAMPLES"]); warmup = int(os.environ["WARMUP"])
outj  = os.environ["OUT_JSON"]

H._bootstrap_gsplat()
spec = importlib.util.spec_from_file_location(modname, so)
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
sys.modules[modname] = ext
H._set_runtime_env()

dev = torch.device("cuda")
result = {}
for scene in scenes:
    fx = H.build_fixture(scene, H.MAX_LONG_SIDE, dev, ext)
    reps_ms = []
    for ri in range(reps):
        for _ in range(warmup):
            H._run_backward(ext, fx)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(samples):
            H._run_backward(ext, fx)
        e.record(); torch.cuda.synchronize()
        reps_ms.append(s.elapsed_time(e) / samples)
    result[scene] = reps_ms
    print("%s %-9s reps_ms=%s" % (os.environ["PROBE_KEY"], scene,
          [round(x,3) for x in reps_ms]), flush=True)
Path(outj).write_text(json.dumps(result))
print("WORKER_DONE", outj)