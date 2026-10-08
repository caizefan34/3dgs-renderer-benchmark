import json
p = "/mnt/storage_pool/liaoyuanjun/acg0_mx/ACG0_MX_DSH/config/mx_target.json"
c = json.load(open(p))
c["allowed_gpu_indices"] = [1, 2, 3, 4, 6, 7]
c["protected_gpu_indices"] = [0, 5]
json.dump(c, open(p, "w"), indent=2)
print("allowed:", c["allowed_gpu_indices"], "protected:", c["protected_gpu_indices"])
