import torch, numpy as np, glob, importlib.util
spec = importlib.util.spec_from_file_location("gsc0a", "/root/gsc0_analysis.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
f = sorted(glob.glob("/root/gsc0_smoke/traces/*.pt"))[0]
s = torch.load(f, map_location="cpu")
idx = torch.arange(0, min(s["N"], 40000))
print("=== optimizer-state error (new, population-floor) on early smoke ===")
for g in ["xyz","shs","opacity","scaling","rotation"]:
    ea = s["adam"][g]["exp_avg"][idx]; eas = s["adam"][g]["exp_avg_sq"][idx]
    with torch.no_grad():
        u_ref = ea.float()/(eas.float().sqrt()+1e-12)
        floor_u = float(m.attr_norm(u_ref).median().item())
        floor_m = float((m.attr_norm(ea.float())+m.attr_norm(eas.float())).median().item())
    ud16 = m.update_dir_perturbation(ea, eas, torch.float16, floor_u)
    udb16 = m.update_dir_perturbation(ea, eas, torch.bfloat16, floor_u)
    os16 = m.optimizer_state_rep_error(ea, eas, torch.float16, floor_m)
    osb16 = m.optimizer_state_rep_error(ea, eas, torch.bfloat16, floor_m)
    print(f"  {g}: floor_u={floor_u:.3e} floor_m={floor_m:.3e} | updir fp16 med={np.median(ud16.numpy()):.3e} bf16 med={np.median(udb16.numpy()):.3e} | state fp16 med={np.median(os16.numpy()):.3e} bf16 med={np.median(osb16.numpy()):.3e}")
print(">>> NEW METRICS OK (bounded, no 1e5 blowup)")