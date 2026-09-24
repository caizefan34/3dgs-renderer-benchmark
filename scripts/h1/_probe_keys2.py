import json
cams = json.load(open("/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json"))
c = cams[0]
print("Keys:", sorted(c.keys()))
print("fx:", c.get("fx"))
print("fy:", c.get("fy"))
print("px:", c.get("px", "NOT FOUND"))
print("py:", c.get("py", "NOT FOUND"))
print("width:", c.get("width"))
print("height:", c.get("height"))
print("img_name:", c.get("img_name"))