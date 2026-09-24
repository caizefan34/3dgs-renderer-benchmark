import json

with open("/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/room_30k/topology_events.json") as f:
    data = json.load(f)

print("Total events:", len(data["events"]))
print("Total clones:", data["total_clones"])
print("Total splits:", data["total_splits"])
print("Total prunes:", data["total_prunes"])
print()

# Show first 5 and last 5 events
for e in data["events"][:5]:
    print(f"  iter {e['iteration']}: N {e['N_before']}->{e['N_after']} clone={e['cloned']} split={e['split']} prune={e['pruned_total']}")
print("  ...")
for e in data["events"][-5:]:
    print(f"  iter {e['iteration']}: N {e['N_before']}->{e['N_after']} clone={e['cloned']} split={e['split']} prune={e['pruned_total']}")

# Opacity reset events (where prune includes screen_size)
screen_prunes = [e for e in data["events"] if e.get("pruned_screen", 0) > 0]
print(f"\nScreen-size prune events: {len(screen_prunes)}")
for e in screen_prunes[:5]:
    print(f"  iter {e['iteration']}: pruned_screen={e['pruned_screen']} pruned_world={e.get('pruned_world', 0)}")

# Peak N
max_n = max(e["N_after"] for e in data["events"])
print(f"\nPeak N: {max_n}")
