#!/usr/bin/env python3
"""Verify signed-by-content manifest for every included file, except itself.

This is an integrity check, NOT authentication of original GPU experiments.
"""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parent.parent
m = json.loads((root/'SHA256_MANIFEST.json').read_text(encoding='utf-8'))
actual = {}
for p in root.rglob('*'):
    if p.is_file() and p.name != 'SHA256_MANIFEST.json' and '__pycache__' not in p.parts:
        actual[str(p.relative_to(root)).replace('\\','/')] = hashlib.sha256(p.read_bytes()).hexdigest()
assert actual == m['files'], {'missing':sorted(set(m['files'])-set(actual)), 'extra': sorted(set(actual)-set(m['files'])), 'mismatch': [k for k in set(actual)&set(m['files']) if actual[k]!=m['files'][k]]}
print(f"MANIFEST_OK {len(actual)}/{len(m['files'])}")
