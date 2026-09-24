import json

d = json.load(open("/mnt/storage_pool/liaoyuanjun/pub_runs/a0_room/results.json"))
def walk(obj, prefix="", depth=0):
    if depth > 2:
        return
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:24]:
            if isinstance(v, (dict, list)):
                print(f"{prefix}{k}: {type(v).__name__}({len(v)})")
                walk(v, prefix + "  ", depth + 1)
            else:
                s = str(v)
                print(f"{prefix}{k}: {s[:80]}")
    elif isinstance(obj, list) and obj:
        print(f"{prefix}[0]: {str(obj[0])[:150]}")
walk(d)
