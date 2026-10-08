import json
p = "h0_run/H0_PARITY.json"
d = json.load(open(p, encoding="utf-8"))
bi = d["binary_identity"]
C0_PIN = "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11"
CORE_PIN = "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"
bi["c0_ext_so_match"] = (bi["c0_ext_so_sha256"] == C0_PIN)
bi["core_so_match"] = (bi["core_so_sha256"] == CORE_PIN)
bi["note"] = "hashes verified against prereg inline pins; match flags recomputed"
json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
print("fixed match flags:", bi["c0_ext_so_match"], bi["core_so_match"])
