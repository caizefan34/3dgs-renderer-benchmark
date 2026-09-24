from PIL import Image
import os
for d in ["/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/datasets/mipnerf360/room/images", "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/datasets/mipnerf360/room/images_4_png"]:
    if os.path.isdir(d):
        f = sorted(os.listdir(d))[0]
        img = Image.open(os.path.join(d, f))
        print(d, f, img.size)