#!/bin/bash
# VOS Phase-4 smoke: 30-iteration traced run on room, GPU 0.
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
mkdir -p "$D"/logs
cd /root/vos_traces
echo "SMOKE_START $(date -u +%FT%TZ)"
$PY /root/vos_traces/vos_trace_driver.py --scene room --iterations 30 --gpu 1 \
  --outdir "$D"/smoke_room30 --no-lpips --final-eval subset \
  > "$D"/logs/smoke_room30.log 2>&1
echo "SMOKE_EXIT=$?"
ls -la "$D"/smoke_room30/ || true
echo "=== meta head ==="
python3 - <<'EOF'
import json
m = json.load(open('/root/vos_traces/smoke_room30/vos_trace_meta.json'))
for k in ('scene','n_captured_training_iters','trace_bin_bytes','packbits_probe_byte',
          'record_errors','WARNING','gpu_name'):
    print(k, '=', m.get(k))
print('records[:5] =', m.get('records', [])[:5])
print('records[-3:] =', m.get('records', [])[-3:])
EOF
echo "=== verify bitmaps decode & cross-check n_visible vs results.json ==="
python3 - <<'EOF'
import json, struct, numpy as np, torch
d = '/root/vos_traces/smoke_room30/'
m = json.load(open(d + 'vos_trace_meta.json'))
blob = open(d + 'vos_trace.bin', 'rb').read()
recs = m['records']
off = 0
bitmaps = []
for (it, N, nb, shd) in recs:
    hdr = struct.unpack_from('<4i', blob, off); off += 16
    assert hdr[0] == it and hdr[1] == N and hdr[2] == nb, (hdr, it, N, nb)
    arr = np.frombuffer(blob, dtype=np.uint8, count=nb, offset=off); off += nb
    bits = np.unpackbits(arr, bitorder='big')[:N].astype(bool)
    bitmaps.append(bits)
    if it in (1, 2, 3, 30):
        print(f'iter {it}: N={N} n_visible(bits)={int(bits.sum())} sh_degree={shd}')
res = json.load(open(d + 'results.json'))
pi = {r['iter']: r for r in res['per_iter']}
for r in res['per_iter'][:3]:
    it = r['iter']
    if it <= len(bitmaps):
        b = bitmaps[it-1]
        print(f"cross-check iter {it}: trainer n_visible={r['n_visible']} "
              f"trace popcount={int(b.sum())} match={r['n_visible']==int(b.sum())} "
              f"N={r['n_gs']}")
print('TRACE_OK' if off == len(blob) else f'TRAKE_MISMATCH off={off} size={len(blob)}')
EOF
echo "SMOKE_DONE $(date -u +%FT%TZ)"
