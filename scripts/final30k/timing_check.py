import json

t = json.load(open("/mnt/storage_pool/liaoyuanjun/final30k_trainer_check/b1a_room_2k/timing.json"))
print({k: t[k] for k in sorted(t) if k != "phase_ms"})
print("phase_ms keys:", sorted(t.get("phase_ms", {}).keys()))
print("phase_ms forward mean:", t.get("phase_ms", {}).get("forward", {}).get("mean_ms"))
