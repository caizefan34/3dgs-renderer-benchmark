import json
d = json.load(open("/tmp/h1_smoke3/train_cam150.json"))
st = d.get("B1_forward_stages_repeated", {})
print("=== per-stage repeated ===")
for k, v in st.items():
    med = v["median_ms"]
    print("  %s: median=%.4f" % (k, med))
stage_sum = sum(v["median_ms"] for v in st.values())
print("  SUM=%.4f" % stage_sum)
t = d.get("B1_forward_decomposed_timing", {})
print("  total_median=%.4f" % t.get("median_ms"))
print("  closure=%.4f" % d.get("timing_closure_B1_forward"))