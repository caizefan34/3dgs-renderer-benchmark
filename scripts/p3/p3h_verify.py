import sys, os

# 1) replicate the PROVEN smoke bootstrap: import gsplat -> WORK via /tmp/p3h_boot
sys.path.insert(0, "/tmp/p3h_boot")
# for importing p3_h_run harness (lives in the run dir)
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun")
import torch

# 2) verify gsplat.rendering._maybe_evaluate_sh resolves from the SAME copy
import gsplat.rendering as R
print("RENDERING_MOD=%s HAS_SHEVAL=%s" % (R.__file__, hasattr(R, "_maybe_evaluate_sh")))
import gsplat.experimental.render.functional.gaussian_inference as GI
print("GI_MOD=%s" % GI.__file__)

import p3_h_run as M
M._set_runtime_env()
dev = torch.device("cuda")
print("GPU_OK=%s" % torch.cuda.get_device_name(0))

# 3) load TWO variants in one process -> collision check (V0 then V1)
for v in ["V0", "V1"]:
    ext = M.load_backend(v, M.so_path_for(v))
    fx = M.build_fixture("room", M.MAX_LONG_SIDE, dev, ext)
    kwargs, out = M._run_backward(ext, fx)
    fin = all(torch.isfinite(o).all().item() for o in out)
    print("VARIANT=%s LOADED_OK=1 BACKWARD_FINITE=%s n_isects=%d" % (v, fin, fx["n_isects"]))
print("VERIFY_DONE")