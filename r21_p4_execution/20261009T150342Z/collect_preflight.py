"""Capture auditable local verification and a read-only MX availability probe."""
import base64
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TASK = REPO/'experiments/r21_opacity_fix'
SSH = 'C:/Windows/System32/OpenSSH/ssh.exe'

def save(name, value):
    (HERE/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def run(name, argv, timeout=90):
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        p = subprocess.run(argv,cwd=REPO,text=True,capture_output=True,timeout=timeout,encoding='utf-8',errors='replace')
        result = dict(argv=argv,started_utc=start,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except subprocess.TimeoutExpired as e:
        result = dict(argv=argv,started_utc=start,exit_code=None,timed_out=True,
                      stdout=(e.stdout or b'').decode('utf-8',errors='replace') if isinstance(e.stdout,bytes) else e.stdout or '',
                      stderr=(e.stderr or b'').decode('utf-8',errors='replace') if isinstance(e.stderr,bytes) else e.stderr or '')
    save(name+'.command.json',result)
    (HERE/(name+'.stdout.log')).write_text(result['stdout'],encoding='utf-8')
    (HERE/(name+'.stderr.log')).write_text(result['stderr'],encoding='utf-8')
    return result

local = run('local_delivery_verify',[sys.executable,str(TASK/'R21_PREPARE_CANDIDATE.py'),'--verify-delivery'])
if local['exit_code'] != 0:
    raise SystemExit('Local delivery verification failed; do not run MX stages')
manifest = json.loads((TASK/'R21_CANDIDATE_MANIFEST.json').read_text(encoding='utf-8'))
git = run('local_git_identity',['git','log','-5','--format=%H %s'])
clean = run('local_candidate_status',['git','status','--porcelain','--','experiments/r21_opacity_fix'])
if git['exit_code'] != 0 or clean['exit_code'] != 0 or clean['stdout'].strip():
    raise SystemExit('Candidate git identity or clean-tree verification failed')
checksum = (TASK/'R21_MANIFEST.sha256').read_text().split()[0]
assert hashlib.sha256((TASK/'R21_CANDIDATE_MANIFEST.json').read_bytes()).hexdigest() == checksum
snapshot = HERE/'candidate_identity'
snapshot.mkdir(exist_ok=False)
for name in ['R21_CANDIDATE_MANIFEST.json','R21_MANIFEST.sha256','R21_OPACITY_FIX.diff','R21_LOCAL_VALIDATION.json']:
    shutil.copyfile(TASK/name,snapshot/name)
save('LOCAL_CANDIDATE_VERIFICATION.json',dict(status='PASS_LOCAL_DELIVERY_IDENTITY_ONLY',
     implementation_commit=manifest['implementation_commit'],validation_commit=manifest['validation_commit'],
     delivery_commit=git['stdout'].split()[0],manifest_sha256=checksum,wrapper_sha256=manifest['R21_wrapper_sha256'],
     payload_files=len(manifest['files']),dependency_files=len(manifest['dependencies']),
     remote_delivery_or_gpu_validation_implied=False))
encoded = base64.b64encode((HERE/'remote_probe.py').read_bytes()).decode()
remote = "python3 -c 'import base64;exec(base64.b64decode(\""+encoded+"\"))'"
argv = [SSH,'-n','-o','BatchMode=yes','-o','ConnectionAttempts=1','-o','ConnectTimeout=8',
        '-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','mx',remote]
result = run('mx_preflight',argv,timeout=90)
if result['exit_code'] != 0:
    raise SystemExit('MX probe did not complete; availability UNKNOWN, not candidate absent')
report = json.loads(result['stdout'])
save('MX_PREFLIGHT.json',report)
print(json.dumps(dict(local_status='PASS_LOCAL_DELIVERY_IDENTITY_ONLY',
                     mx_candidate=report['candidate_availability'],
                     markers=report['found_candidate_markers'],scan_errors=report['scan_errors'],
                     remote_root=report['output_root'],local_evidence=str(HERE)),indent=2))
