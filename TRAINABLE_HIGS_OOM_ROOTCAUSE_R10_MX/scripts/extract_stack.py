#!/usr/bin/env python3
"""R10: Extract full OOM stack trace"""
import pickle
import json

print("=== R10: Extracting Full OOM Stack Trace ===\n")

snapshot_path = "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OOM_ROOTCAUSE_R10_MX/raw/pytorch_oom_snapshot.pickle"

with open(snapshot_path, 'rb') as f:
    snapshot = pickle.load(f)

device_traces = snapshot.get('device_traces', [])

for device_idx, traces in enumerate(device_traces):
    if not isinstance(traces, list):
        continue
    
    for event in traces:
        if event.get('action') == 'oom':
            print(f"OOM on device {device_idx}")
            print(f"Requested: {event.get('size', 0)} bytes")
            print(f"\nFull stack trace ({len(event.get('frames', []))} frames):")
            
            frames = event.get('frames', [])
            for i, frame in enumerate(frames):
                filename = frame.get('filename', 'unknown')
                line = frame.get('line', 0)
                name = frame.get('name', 'unknown')
                print(f"{i+1}. {name}")
                print(f"   {filename}:{line}")
            
            # Save full stack
            output = {
                'device': device_idx,
                'size_bytes': event.get('size', 0),
                'size_gb': event.get('size', 0) / 1e9,
                'frames': frames
            }
            
            output_path = "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OOM_ROOTCAUSE_R10_MX/raw/oom_full_stack.json"
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\n✓ Full stack saved: {output_path}")
