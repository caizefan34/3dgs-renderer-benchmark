import json

d = json.load(open("/tmp/h1_smoke/train_cam150.json"))
print("=== timing closure ===")
print("timing_closure_B1_forward:", d.get("timing_closure_B1_forward"))
print("B1_forward_total_ms:", d.get("B1_forward_total_ms"))
print("B1_renderer_total_ms:", d.get("B1_renderer_total_ms"))
print("B2_forward_total_ms:", d.get("B2_forward_total_ms"))
print("B2_renderer_total_ms:", d.get("B2_renderer_total_ms"))
print("speed_ratio_B2_over_B1:", d.get("speed_ratio_B2_over_B1"))
print("delta_forward_ms:", d.get("delta_forward_ms"))
print("delta_backward_ms:", d.get("delta_backward_ms"))
print("delta_renderer_total_ms:", d.get("delta_renderer_total_ms"))

print("\n=== forward correctness ===")
fc = d.get("forward_correctness", {})
for k, v in fc.items():
    print(f"  {k}: {v}")

print("\n=== B1 stages (single cold pass) ===")
st = d.get("B1_forward_stages_single", {})
for k, v in st.items():
    print(f"  {k}: {v:.4f}")

print("\n=== B1 workload ===")
wl = d.get("B1_workload", {})
for k, v in wl.items():
    print(f"  {k}: {v}")

print("\n=== B2 workload ===")
wl2 = d.get("B2_workload", {})
for k, v in wl2.items():
    print(f"  {k}: {v}")

print("\n=== B2 metadata (scalar) ===")
md = d.get("B2_metadata", {})
for k, v in md.items():
    if isinstance(v, (int, float, str, bool)):
        print(f"  {k}: {v}")
    elif isinstance(v, dict):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: {type(v).__name__} len={len(v) if hasattr(v, '__len__') else '?'}")