try:
    import PIL; print("PIL", PIL.__version__)
except Exception as e: print("no PIL", e)
try:
    import imageio; print("imageio", imageio.__version__)
except Exception as e: print("no imageio", e)
try:
    import matplotlib; print("mpl", matplotlib.__version__)
except Exception as e: print("no mpl", e)
import numpy as np, zlib, struct
def write_png(path, arr):
    # arr uint8 HxWx3
    h,w,_=arr.shape
    raw=b"".join(b"\x00"+arr[i].tobytes() for i in range(h))
    def chunk(t,d):
        c=t+d
        return struct.pack(">I",len(d))+c+struct.pack(">I",zlib.crc32(c)&0xffffffff)
    sig=b"\x89PNG\r\n\x1a\n"
    ihdr=struct.pack(">IIBBBBB",w,h,8,2,0,0,0)
    idat=zlib.compress(raw,6)
    with open(path,"wb") as f:
        f.write(sig+chunk(b"IHDR",ihdr)+chunk(b"IDAT",idat)+chunk(b"IEND",b""))
print("manual png writer ready")
