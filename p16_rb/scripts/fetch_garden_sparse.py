#!/usr/bin/env python3
"""Range-fetch garden sparse files from the official gresearch 360_v2.zip.

Non-destructive: writes only into OUT dir; verifies CRC32 + sizes; never touches
the repo dataset. Lists CD entries first, then fetches garden/sparse/* members
plus one sample image for provenance byte-comparison.
"""
import json
import os
import struct
import sys
import urllib.request
import zlib
import hashlib

URL = 'https://storage.googleapis.com/gresearch/refraw360/360_v2.zip'
OUT = '/root/epic05_p16rb/garden_repair/official'
os.makedirs(OUT, exist_ok=True)


def fetch_range(a, b, tries=6):
    if b < a:
        return b''
    req = urllib.request.Request(URL, headers={'Range': f'bytes={a}-{b}'})
    last = None
    for t in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != b - a + 1:
                raise IOError(f'short read {len(data)} != {b - a + 1}')
            return data
        except Exception as e:  # noqa
            last = e
            print(f'retry {t}: {e}', flush=True)
    raise IOError(f'range fetch failed {a}-{b}: {last}')


req = urllib.request.Request(URL, method='HEAD')
with urllib.request.urlopen(req, timeout=60) as r:
    size = int(r.headers['Content-Length'])
print('zip size', size, flush=True)

tail = fetch_range(max(0, size - 262144), size - 1)
i = tail.rfind(b'PK\x05\x06')
assert i >= 0, 'EOCD not found'
n_entries = struct.unpack('<H', tail[i + 10:i + 12])[0]
cd_size = struct.unpack('<I', tail[i + 12:i + 16])[0]
cd_off = struct.unpack('<I', tail[i + 16:i + 20])[0]
zip64 = False
if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_entries == 0xFFFF:
    zip64 = True
    j = tail.rfind(b'PK\x06\x07')
    assert j >= 0, 'zip64 locator not found'
    z64_loc = struct.unpack('<Q', tail[j + 8:j + 16])[0]
    z64 = fetch_range(z64_loc, z64_loc + 55)
    assert z64[:4] == b'PK\x06\x06'
    n_entries = struct.unpack('<Q', z64[32:40])[0]
    cd_size = struct.unpack('<Q', z64[40:48])[0]
    cd_off = struct.unpack('<Q', z64[48:56])[0]
print('cd_off', cd_off, 'cd_size', cd_size, 'n_entries', n_entries, 'zip64', zip64, flush=True)

cd = fetch_range(cd_off, cd_off + cd_size - 1)
entries = []
p = 0
while p < len(cd) - 4 and cd[p:p + 4] == b'PK\x01\x02':
    (ver, ver2, flags, method, mt, md, crc, csize, usize,
     fnl, el, cl, disk, iat, eat, lho) = struct.unpack('<HHHHHHIIIHHHHHII', cd[p + 4:p + 46])
    fn = cd[p + 46:p + 46 + fnl].decode('utf-8', 'replace')
    extra = cd[p + 46 + fnl:p + 46 + fnl + el]
    p += 46 + fnl + el + cl
    if (csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or lho == 0xFFFFFFFF):
        q = 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack('<HH', extra[q:q + 4])
            if hid == 0x0001:
                vals = extra[q + 4:q + 4 + hsz]
                off = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack('<Q', vals[off:off + 8])[0]; off += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack('<Q', vals[off:off + 8])[0]; off += 8
                if lho == 0xFFFFFFFF:
                    lho = struct.unpack('<Q', vals[off:off + 8])[0]; off += 8
                break
            q += 4 + hsz
    entries.append({'name': fn, 'method': method, 'csize': csize, 'usize': usize,
                    'crc': crc, 'lho': lho, 'flags': flags})

print('parsed entries', len(entries), flush=True)
tops = {}
for e in entries:
    top = e['name'].split('/')[0]
    tops[top] = tops.get(top, 0) + 1
print('top-level:', json.dumps(tops, indent=1), flush=True)

targets = [e for e in entries if e['name'].startswith('garden/')]
sparse = [e for e in targets if '/sparse/' in e['name']]
imgs = [e for e in targets if '/images/' in e['name']]
print('garden entries:', len(targets), 'sparse:', len(sparse), 'images:', len(imgs), flush=True)
for e in sparse:
    print('  SPARSE', e['name'], 'csize', e['csize'], 'usize', e['usize'], 'method', e['method'], flush=True)

todo = [e for e in sparse if not e['name'].endswith('/')]
imgs = [e for e in imgs if not e['name'].endswith('/')]
if imgs:
    todo.append(sorted(imgs, key=lambda x: x['name'])[0])

manifest = []
for e in todo:
    rel = e['name'].replace('garden/', '')
    dst = os.path.join(OUT, rel)
    if os.path.exists(dst) and os.path.getsize(dst) == e['usize']:
        with open(dst, 'rb') as fh:
            blob = fh.read()
        if zlib.crc32(blob) & 0xFFFFFFFF == e['crc']:
            print('SKIP', dst, '(already present, crc ok)', flush=True)
            manifest.append({'name': e['name'], 'path': dst, 'bytes': len(blob),
                             'crc32': format(e['crc'], '08x'),
                             'sha256': hashlib.sha256(blob).hexdigest(), 'method': e['method']})
            continue
    lh = fetch_range(e['lho'], e['lho'] + 29)
    assert lh[:4] == b'PK\x03\x04', f'bad local header {e["name"]}'
    fnl2, el2 = struct.unpack('<HH', lh[26:30])
    data_start = e['lho'] + 30 + fnl2 + el2
    raw = fetch_range(data_start, data_start + e['csize'] - 1)
    if e['method'] == 0:
        blob = raw
    elif e['method'] == 8:
        blob = zlib.decompressobj(-15).decompress(raw)
    else:
        raise ValueError(f'unsupported method {e["method"]} for {e["name"]}')
    assert len(blob) == e['usize'], f'size mismatch {e["name"]}: {len(blob)} vs {e["usize"]}'
    assert zlib.crc32(blob) & 0xFFFFFFFF == e['crc'], f'crc mismatch {e["name"]}'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as fh:
        fh.write(blob)
    sha = hashlib.sha256(blob).hexdigest()
    manifest.append({'name': e['name'], 'path': dst, 'bytes': len(blob),
                     'crc32': format(e['crc'], '08x'), 'sha256': sha, 'method': e['method']})
    print('OK', dst, len(blob), sha[:16], flush=True)

with open(os.path.join(OUT, 'MANIFEST_garden_official.json'), 'w') as fh:
    json.dump({'source': URL, 'zip_bytes': size, 'members': manifest}, fh, indent=1)
print('DONE', flush=True)
