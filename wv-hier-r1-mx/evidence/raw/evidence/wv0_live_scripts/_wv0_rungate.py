import torch
g=torch.load("/dev/shm/wv0_out/WV0_GATE_EVALUATION.pt", weights_only=False)
gr=g["gate_results"]
print("keys:", list(list(gr.values())[0].keys()))
for k,v in gr.items():
    print(f"cand G{k}: gateA_pass={v.get('gateA_pass')} gateB_pass={v.get('gateB_pass')} "
          f"n_scenes_pass={v.get('n_scenes_pass')}")
    print("  gateB_1cos_delta:", v.get('gateB_1cos_delta'))
    print("  passing_scenes:", {sc: vv.get('n_pass') for sc,vv in v.get('passing_scenes',{}).items()})
