import json

path = r"C:\Users\36570\3dgs-renderer-benchmark\reports\r6\r6-profile-results-v2.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== Top-level ===", list(data.keys()))
print()
print("=== r6_b_correctness ===")
print(json.dumps(data.get("r6_b_correctness", {}), indent=2)[:3000])
print()
print("=== r6_b_benchmark present? ===", "r6_b_benchmark" in data)
if "r6_b_benchmark" in data:
    print(json.dumps(data["r6_b_benchmark"], indent=2)[:3000])
