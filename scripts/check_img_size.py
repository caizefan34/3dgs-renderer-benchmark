#!/usr/bin/env python3
from PIL import Image
im = Image.open("/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/datasets/mipnerf360/room/images_4_png/DSCF4667.png")
print("size:", im.size)
im2 = Image.open("/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/datasets/mipnerf360/room/images/DSCF4667.JPG")
print("full size:", im2.size)
