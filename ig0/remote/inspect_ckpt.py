import sys, torch, platform, hashlib
print("hostname", platform.node())
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("CUDA_VISIBLE_DEVICES", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))

CKPTS = {
    "room@10000": "/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_10000.pt",
    "room@15000": "/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_15000.pt",
    "bicycle@10000": "/root/epic05_n1/runs/n1snap_c0f_s42_bicycle/ckpt_10000.pt",
    "bicycle@15000": "/root/epic05_n1/runs/n1snap_c0f_s42_bicycle/ckpt_15000.pt",
}
for name, path in CKPTS.items():
    print("=" * 60)
    print(name, path)
    try:
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception as e:
        print("  LOAD ERROR:", repr(e))
        continue
    if isinstance(sd, dict):
        print("  top-level keys:", list(sd.keys()))
        for k in sd:
            v = sd[k]
            if isinstance(v, dict):
                sub = list(v.keys())
                print("  [%s] dict keys(%d): %s" % (k, len(sub), sub[:12]))
                # optimizer state?
                if any("state" in str(s).lower() for s in sub):
                    print("    -> has state-like keys")
            elif torch.is_tensor(v):
                print("  [%s] tensor %s %s" % (k, tuple(v.shape), v.dtype))
            else:
                print("  [%s] %s" % (k, type(v).__name__))
