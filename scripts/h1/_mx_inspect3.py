import json

d = json.load(open("/tmp/h1_smoke2/train_cam150.json"))
print("=== B1_forward_stages_repeated ===")
st = d.get("B1_forward_stages_repeated", {})
for k, v in st.items():
    print(f"  {k}: median={v['median_ms']:.4f} mean={v['mean_ms']:.4f} std={v['std_ms']:.4f}")

print("\n=== B1_forward_stages_single (cold) ===")
st1 = d.get("B1_forward_stages_single", {})
for k, v in st1.items():
    print(f"  {k}: {v:.4f}")

print("\n=== timing_closure_B1_forward ===")
print(f"  {d.get('timing_closure_B1_forward')}")

print("\n=== B1_forward_decomposed_timing ===")
t = d.get("B1_forward_decomposed_timing", {})
print(f"  median={t.get('median_ms')} mean={t.get('mean_ms')}")

print("\n=== B1_forward_autograd_timing ===")
t2 = d.get("B1_forward_autograd_timing', {}")
print(f"  median={t2.get('median_ms')}")

print("\n=== B1/B2 total times ===")
for k in ["B1_forward_total_ms", "B1_renderer_total_ms", "B2_forward_total_ms", "B2_renderer_total_ms",
          "B1_backward_total_ms", "B2_backward_total_ms", "delta_forward_ms", "delta_backward_ms",
          "delta_renderer_total_ms", "speed_ratio_B2_over_B1"]:
    print(f"  {k}: {d.get(k)}")