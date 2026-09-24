"""Add garden seed-44 pair for 3-seed confirmation of the accutile garden
quality effect (s42: -0.642, s43: -0.806 dB -> is s43 magnitude stable?).
Insert BEFORE the b1/b0 expansions so it runs next. Idempotent by key.
"""
import json
import os

QF = "/mnt/storage_pool/liaoyuanjun/pubphase/pub_queue.json"
q = json.load(open(QF))
have = {(e["key"], e["scene"]) for e in q}
add = [
    {"key": "b1as44", "arm": "b1a", "scene": "garden", "extra": ["--seed", "44"]},
    {"key": "b1s44", "arm": "b1", "scene": "garden", "extra": ["--seed", "44"]},
]
missing = [e for e in add if (e["key"], e["scene"]) not in have]
if missing:
    # insert right after the Stage B block (before b1 x10 expansion)
    idx = next(i for i, e in enumerate(q) if e["key"] == "b1" and e["scene"] == "bonsai")
    q[idx:idx] = missing
    with open(QF + ".tmp", "w") as f:
        json.dump(q, f, indent=1)
    os.replace(QF + ".tmp", QF)
    print(f"INSERTED {len(missing)} garden s44 entries at position {idx}; queue now {len(q)} entries")
else:
    print("garden s44 pair already queued")
