#!/usr/bin/env python3
"""Stage-0 read-only repository identity and evidence inventory.

Never writes to --repo. Only stdlib is used. Results MUST be interpreted
manually; file name matches do not confirm validity/completion.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess

WATCH = [
    'PUBLICATION_CONVERGENCE_PACK_V2_1/PAPER1_REPORT.md',
    'PUBLICATION_CONVERGENCE_PACK_V2_1/PAPER1_OPEN_GAPS.json',
    'PUBLICATION_CONVERGENCE_PACK_V2_1/PAPER1_MAIN_TABLE_DRAFT.json',
    'PAPER0_HARDENED/PAPER0_HARDENED_EVIDENCE_MATRIX.json',
    'CLOSED_DIRECTIONS_20260930.md',
    'reports/bes0_aabb_reject_measured.md',
]
HINTS = ('tfc', 'pbr0', 'pub0', 'accutile', 'footprint', 'ellipse')


def sha_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def git(repo, *args):
    try:
        p = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=12, check=False)
        return {'returncode': p.returncode, 'stdout': p.stdout.strip()[:100000], 'stderr': p.stderr.strip()[:2000]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'error': str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    repo = pathlib.Path(a.repo).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser().resolve()
    if not repo.is_dir():
        ap.error('--repo must be an existing repository directory')
    # Prevent accidental writing into the source tree, even in a nested subdirectory.
    if out == repo or repo in out.parents:
        ap.error('--out MUST be outside --repo')
    # Avoid output directory being an alias to repository or its parent.
    out.mkdir(parents=True, exist_ok=True)
    git_meta = {k: git(repo, *v) for k,v in {
        'head': ['rev-parse','HEAD'], 'branch': ['branch','--show-current'],
        'status': ['status','--porcelain=v1'], 'submodule': ['submodule','status']
    }.items()}
    watch = {}
    for rel in WATCH:
        p = repo / rel
        watch[rel] = {'present':p.is_file(), 'sha256':sha_file(p) if p.is_file() else None}
    matches=[]
    # Filenames-only lookup, cap; not a content interpretation or exhaustive source search.
    excluded={'.git', '__pycache__', 'node_modules', '.venv', 'build'}
    for root,dirs,files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in excluded]
        for fn in files:
            if any(h in fn.lower() for h in HINTS):
                p=pathlib.Path(root)/fn
                try:
                    matches.append({'relative_path': str(p.relative_to(repo)), 'bytes':p.stat().st_size})
                except OSError: pass
            if len(matches)>=2500: break
        if len(matches)>=2500: break
    data = {
        'schema':'tfc-p0-inventory/v1', 'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'READ_ONLY', 'repo':str(repo), 'out':str(out),
        'git':git_meta, 'watched_files':watch,
        'filename_candidates':matches, 'filename_results_truncated':len(matches)>=2500,
        'tier_a_tfc_fc0_verified': False, 'novelty_verified':False,
        'caveat':'Identity inventory only, not MX measurement, not proof of novel mechanism.'
    }
    (out/'TFC_P0_SOURCE_INVENTORY.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'saved':str(out/'TFC_P0_SOURCE_INVENTORY.json'),'found':sum(x['present'] for x in watch.values()),'watched':len(watch),'candidates':len(matches),'read_only':True}))

if __name__ == '__main__': main()
