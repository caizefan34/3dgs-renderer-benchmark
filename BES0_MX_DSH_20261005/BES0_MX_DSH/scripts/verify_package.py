#!/usr/bin/env python3
"""Static integrity/syntax check only. Safe locally; does not import experiment code."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = root/'MANIFEST.sha256'
failures=[]
listed=set()
for line in manifest.read_text().splitlines():
    digest, name = line.split('  ',1)
    if name in listed:
        failures.append(name+': duplicate manifest entry')
    listed.add(name)
    path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failures.append(name+': missing/unsafe')
    elif hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        failures.append(name+': hash differs')
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
        and p.name!='MANIFEST.sha256' and '__pycache__' not in p.parts}
if actual!=listed:
    failures.append('manifest does not cover exactly the package files (added code must be archived with updated manifest)')
for path in root.rglob('*.py'):
    ast.parse(path.read_text(),filename=str(path))
for path in root.rglob('*.json'):
    json.loads(path.read_text())
print(json.dumps({'static_integrity':'PASS' if not failures else 'FAIL','failures':failures,
                  'scientific_experiments':'NOT_RUN'},indent=2))
raise SystemExit(bool(failures))
