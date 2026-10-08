import json
s = json.load(open("/root/gsc0_out/GSC0_SUMMARY.json"))
o = json.load(open("/root/gsc0_out/GSC0_ERROR_ORACLE.json"))
for scene in ["room"]:
    r = s["per_scene"][scene]; oc = o[scene]
    print("=====", scene, "=====")
    print("id_match:", r["id_match"])
    print("gradient_magnitude:", json.dumps(r["gradient_magnitude"], indent=1)[:600])
    print("visibility:", json.dumps(r["visibility_mature_window"], indent=1)[:400])
    print("memory_model:", json.dumps(oc["memory_model"], indent=1))
    print("stable_subset_size:", oc["stable_subset_size"])
    print("gateB_detail:", json.dumps(r.get("gateB_detail"), indent=1))
    print("early->mature (room) xyz median:", r["stability_early_to_mature"]["xyz"]["median"])
# also print gate block
print("===== GATE =====")
print(json.dumps(s["gate"], indent=1))