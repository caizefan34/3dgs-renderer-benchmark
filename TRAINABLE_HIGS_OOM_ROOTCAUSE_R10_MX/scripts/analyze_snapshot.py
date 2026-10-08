#!/usr/bin/env python3
"""R10: Analyze PyTorch OOM snapshot"""
import pickle
import json

print("=== R10: Analyzing PyTorch OOM Snapshot ===\n")

snapshot_path = "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OOM_ROOTCAUSE_R10_MX/raw/pytorch_oom_snapshot.pickle"

print(f"Loading snapshot: {snapshot_path}")
with open(snapshot_path, 'rb') as f:
    snapshot = pickle.load(f)

print(f"Snapshot keys: {snapshot.keys()}")
print(f"Device traces: {len(snapshot.get('device_traces', []))} devices")

# Find OOM events
oom_events = []
device_traces = snapshot.get('device_traces', [])

for device_idx, traces in enumerate(device_traces):
    if not isinstance(traces, list):
        continue
    print(f"\nDevice {device_idx}: {len(traces)} events")

    for event in traces:
        if event.get('action') == 'oom':
            oom_events.append({
                'device': device_idx,
                'size': event.get('size', 0),
                'stream': event.get('stream', -1),
                'frames': event.get('frames', [])
            })

print(f"\n=== OOM Events: {len(oom_events)} ===")

if oom_events:
    for i, oom in enumerate(oom_events):
        print(f"\nOOM Event {i+1}:")
        print(f"  Device: {oom['device']}")
        print(f"  Requested: {oom['size'] / 1e9:.2f} GB ({oom['size'] / 1024**3:.2f} GiB)")
        print(f"  Stream: {oom['stream']}")
        print(f"  Stack frames: {len(oom['frames'])}")
        
        # Print top stack frames
        for j, frame in enumerate(oom['frames'][:10]):
            print(f"    {j+1}. {frame.get('filename', 'unknown')}:{frame.get('line', 0)} in {frame.get('name', 'unknown')}")
else:
    print("No OOM events found in snapshot")
    print("\nPossible reasons:")
    print("1. OOM occurred outside PyTorch allocator")
    print("2. Snapshot range limited")
    print("3. Native CUDA allocation (cudaMalloc)")

# Save OOM events
if oom_events:
    output_path = "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OOM_ROOTCAUSE_R10_MX/raw/oom_events.json"
    with open(output_path, 'w') as f:
        json.dump(oom_events, f, indent=2)
    print(f"\n✓ OOM events saved: {output_path}")
