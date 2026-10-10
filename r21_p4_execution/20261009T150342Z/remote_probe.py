"""Read-only MX candidate-availability audit. Does not import CUDA or deploy code."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASE = Path('/mnt/storage_pool/liaoyuanjun')
REPO = Path('/home/liaoyuanjun/3dgs-renderer-benchmark')
OUT = BASE/'TRAINABLE_HIGS_OPACITY_FIX_R21_P4_MX'
COMMITS = ['4d10249f835fce15ec8c67a21d99358508b01cde',
           'bdd3084e7124f887e17aa7855be0c0ee81c7d2bf',
           'ddb2354f299aa10ae1567309111df66a586061ae']

def run(argv):
    p = subprocess.run(argv, text=True, capture_output=True, timeout=30)
    return dict(argv=argv, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)

def inspect(path):
    result = dict(path=str(path), exists=path.exists(), is_file=path.is_file(), is_dir=path.is_dir())
    if path.is_file():
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024), b''):
                h.update(chunk)
        result.update(size=path.stat().st_size, sha256=h.hexdigest())
    return result

report = dict(schema='r21.p4.mx-preflight.v1', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              scope='Read-only inspection; no GPU kernels, compilation, training, deployment or checkpoint writes',
              output_root=inspect(OUT), commands=[], paths=[], found_candidate_markers=[], scan_errors=[])
report['commands'].append(run(['hostname']))
report['commands'].append(run(['git','-C',str(REPO),'rev-parse','HEAD']))
report['commands'].append(run(['git','-C',str(REPO),'branch','--show-current']))
for commit in COMMITS:
    report['commands'].append(run(['git','-C',str(REPO),'cat-file','-e',commit+'^{commit}']))
report['commands'].append(run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu','--format=csv']))
report['paths'].append(inspect(REPO/'experiments/r21_opacity_fix/R21_CANDIDATE_MANIFEST.json'))
report['paths'].append(inspect(OUT/'gsplat-higs-r21-opacity/R21_BUILD_IDENTITY.json'))
report['paths'].append(inspect(OUT/'gsplat-higs-r21-opacity/gsplat/experimental/render/functional/gaussian_inference.py'))
roots = [p for p in BASE.iterdir() if p.is_dir() and 'r21' in p.name.lower()]
roots += [REPO/'experiments/r21_opacity_fix']
report['scan_roots'] = [str(p) for p in roots]
report['scan_depth_limit'] = 4
for root in roots:
    if not root.is_dir():
        continue
    for directory, dirs, files in os.walk(root, onerror=lambda e:report['scan_errors'].append(str(e))):
        current = Path(directory)
        if len(current.relative_to(root).parts) >= 4:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in {'.git','__pycache__','build'}]
        for name in ['R21_CANDIDATE_MANIFEST.json','R21_BUILD_IDENTITY.json']:
            if name in files:
                report['found_candidate_markers'].append(inspect(current/name))
r20 = BASE/'TRAINABLE_HIGS_ABSGRAD_PARITY_R20_MX/gsplat-higs-r20'
frozen = Path('/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')
for root, relatives in [(r20,['R20_BUILD_IDENTITY.json','gsplat/csrc.so','gsplat/experimental/render/kernels/csrc.so',
                              'gsplat/experimental/render/functional/gaussian_inference.py']),
                        (frozen,['gsplat/csrc.so'])]:
    for relative in relatives:
        report['paths'].append(inspect(root/relative))
report['candidate_availability'] = ('CANDIDATE_MARKER_FOUND_REQUIRES_VERIFICATION' if report['found_candidate_markers']
                                    else 'BLOCKED_CANDIDATE_NOT_AVAILABLE')
print(json.dumps(report, indent=2))
