import torch, numpy as np, glob, os, sys
sys.path.insert(0, "/root/gsc0_p0")
import importlib.util
spec = importlib.util.spec_from_file_location("gsc0a", "/root/gsc0_analysis.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
f = sorted(glob.glob("/root/gsc0_smoke/traces/*.pt"))[0]
s = torch.load(f, map_location="cpu")
print("phase", s["phase"], "iter", s["iteration"], "N", s["N"], "shdeg", s["active_sh_degree"])
slots = m.n_active_slots(int(s["active_sh_degree"]))
print("active slots", slots)
# rep error on a subset for speed
idx = torch.arange(0, min(s["N"], 20000))
for a in ["xyz","scaling","rotation","opacity"]:
    x = s[a][idx]
    r16 = m.cast_rel_error(x, torch.float16)
    rb16 = m.cast_rel_error(x, torch.bfloat16)
    of16, nf16, tot16 = m.cast_overflow_frac(x, torch.float16)
    print(f"  {a}: fp16 median={np.median(r16.numpy()):.3e} bf16 median={np.median(rb16.numpy()):.3e} fp16_overflow={of16:.2e}")
x = s["shs"][idx][:, :slots, :]
print(f"  shs({slots}): fp16 median={np.median(m.cast_rel_error(x, torch.float16).numpy()):.3e} bf16 median={np.median(m.cast_rel_error(x, torch.bfloat16).numpy()):.3e}")
# adam perturbation
for g in ["xyz","shs","opacity","scaling","rotation"]:
    ea = s["adam"][g]["exp_avg"][idx]; eas = s["adam"][g]["exp_avg_sq"][idx]
    p16 = m.adam_dir_perturbation(ea, eas, torch.float16)
    pb16 = m.adam_dir_perturbation(ea, eas, torch.bfloat16)
    print(f"  adam[{g}]: fp16 median={np.median(p16.numpy()):.3e} bf16 median={np.median(pb16.numpy()):.3e}")
# match_by_id self-match
common, ia, ib = m.match_by_id(s, s)
print("self-match common", common.size, "expect", s["N"], "frozen", common.size==s["N"])
print(">>> ORACLE MATH OK")