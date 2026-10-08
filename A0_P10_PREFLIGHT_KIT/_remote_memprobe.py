import json, os
def rd(p):
    if os.path.exists(p):
        try:
            return open(p).read()
        except Exception as e:
            return f"ERR {e}"
    return "(missing)"
print("=== n1snap room memory.json ==="); print(rd("/root/epic05_n1/runs/n1snap_c0f_s42_room/memory.json"))
print("=== n1snap room final_status.json ==="); print(rd("/root/epic05_n1/runs/n1snap_c0f_s42_room/final_status.json"))
print("=== n1snap room timing.json ==="); print(rd("/root/epic05_n1/runs/n1snap_c0f_s42_room/timing.json"))
print("=== n1snap room stdout.log (tail) ==="); print(rd("/root/epic05_n1/runs/n1snap_c0f_s42_room/stdout.log")[-2000:])
print("=== Q0 room_25000 run.log (head 40) ==="); print(rd("/root/epic05_moonshot/q0/room_25000/run.log")[:1600])
print("=== Q0 room_25000 run.log: OOM/mem lines ===")
txt = rd("/root/epic05_moonshot/q0/room_25000/run.log")
for ln in txt.splitlines():
    if any(k in ln for k in ("OOM","memory","GiB","GB ","CUDA_VISIBLE","intersect","N=","num_points","gaussian")):
        print("  |", ln[:160])
