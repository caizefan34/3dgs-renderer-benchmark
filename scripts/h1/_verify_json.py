import json
d = json.load(open("/mnt/storage_pool/3dgs-renderer-benchmark/repo/artifacts/h1-clean-profile/backward_coverage_sanity.json"))
print("Valid JSON")
print("overall_verdict:", d["overall_verdict"])
print("backward_correctness_closed:", d["backward_correctness_closed"])
print("diagnosis:", d["diagnosis_previous_6_gaussians"]["classification"])
print("root_cause:", d["diagnosis_previous_6_gaussians"]["root_cause"])
print("cameras:", list(d["cameras"].keys()))
for ci in d["cameras"]:
    c = d["cameras"][ci]
    print("  cam %s: classification=%s, P1 means nz=%d" % (ci, c["classification"], c["gradient_support"]["P1"]["means"]["N_nonzero_gaussians"]))