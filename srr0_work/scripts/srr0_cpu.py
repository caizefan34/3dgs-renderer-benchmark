import json
d = json.load(open('/root/srr0_room_profile.json'))
# top CPU self time events
rows = []
for ev in d['kernels']:
    rows.append((ev.get('self_cpu_time_total',0.0), ev['name']))
rows.sort(reverse=True)
print("TOP CPU self-time (us):")
for t,n in rows[:12]:
    print(f"  {t:.0f}us {n[:80]}")
# aggregate
tot_cpu = sum(ev.get('self_cpu_time_total',0.0) for ev in d['kernels'])
tot_gpu = sum(ev['self_cuda_us'] for ev in d['kernels'] if not ev['name'].startswith('_Rasterize'))
print(f"sum self_cpu_us={tot_cpu:.0f}  sum self_gpu_us(leaf)={tot_gpu:.0f}")
