import json
for ci in (12, 38, 64, 90, 116, 142, 168, 194, 220, 246, 272, 298):
    u = json.load(open(f"/root/epic05_p17/P17_UNITS/room/camera_{ci}.json"))
    print(ci, u["detail"]["cost_by_d"], "d2+=", u["detail"]["n_assigned_d2plus"])
