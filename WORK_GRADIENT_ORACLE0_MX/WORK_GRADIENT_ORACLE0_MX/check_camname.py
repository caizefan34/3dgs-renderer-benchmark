import json, os
c=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
print("cam0 img_name:", repr(c[0]["img_name"]))
print("cam1 img_name:", repr(c[1]["img_name"]))
d="/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_images"
print("eval_images exists:", os.path.isdir(d))
if os.path.isdir(d):
    fs=sorted(os.listdir(d))[:5]
    print("first files:", fs)
