import os, glob
def rd(p):
    try: return open(p).read()
    except Exception as e: return f"ERR {e}"
base = "/root/epic05_moonshot/q0/room_25000"
print("=== Q0 room_25000 files ===")
for f in sorted(os.listdir(base)):
    print("  ", f, os.path.getsize(os.path.join(base,f)))
for fn in ("cell.log","fin.log","run.log","Q0_STATUS_room_25000.json","Q0_CHECKPOINT_IDENTITY_room_25000.json"):
    p = os.path.join(base, fn)
    if os.path.exists(p):
        print(f"=== {fn} ===")
        print(rd(p)[:2500])
