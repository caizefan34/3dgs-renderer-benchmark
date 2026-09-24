import json
d = json.load(open("/mnt/storage_pool/3dgs-renderer-benchmark/repo/artifacts/h1-clean-profile/backward_coverage_sanity.json"))
c0 = d["cameras"]["0"]
for path in ["P1", "P2", "P3"]:
    s = c0["gradient_support"][path]["means"]
    print("%s means: nz_gaussians=%d, exact_zero_elements=%d, total_elements=%d" % (
        path, s["N_nonzero_gaussians"], s["N_exact_zero_elements"], s["N_total"]*3))
    s = c0["gradient_support"][path]["opacities"]
    print("%s opacities: nz_gaussians=%d, exact_zero_elements=%d, total_elements=%d" % (
        path, s["N_nonzero_gaussians"], s["N_exact_zero_elements"], s["N_total"]))
print()
print("L1 loss cam0:", c0.get("loss_type"))
print("N_visible (B2):", c0["N_visible"])
print("N_total:", c0["N_total"])