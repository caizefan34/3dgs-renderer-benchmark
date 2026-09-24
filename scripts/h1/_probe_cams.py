import json
cams = json.load(open("/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json"))
print("n_cameras:", len(cams))
for i in [0, len(cams)//2, len(cams)-1]:
    c = cams[i]
    print("  cam[%d]: img_name=%s, width=%s, height=%s, fx=%s" % (i, c.get("img_name","?"), c.get("width","?"), c.get("height","?"), c.get("fx","?")))