"""Resume evidence sealing after a transport-only failure; never rerun GPU work."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT='/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OPACITY_FIX_R21_P4_MX'
attempt=sys.argv[1]
assert attempt.isdecimal()
logs={name:json.loads((HERE/name).read_text()) for name in
      ['mx_upload_preflight.command.json','mx_finalize_evidence.command.json']}
encoded_logs=base64.b64encode(json.dumps(logs,indent=2).encode()).decode()
encoded_code=base64.b64encode((HERE/'finalize_mx_evidence.py').read_bytes()).decode()
digest=hashlib.sha256((HERE/'preflight_payload.tar').read_bytes()).hexdigest()
code=f'''import base64,hashlib,json,sys
from pathlib import Path
r=Path('{ROOT}')
p=r/'TRANSPORT_LOG.json'
if not p.exists():
    with p.open('xb') as f:f.write(base64.b64decode('{encoded_logs}'))
a=r/'r21_p4_return.tar.gz'
if a.exists():
    print(json.dumps(dict(status=json.loads((r/'RUN_STATUS.json').read_text())['status'],archive=str(a),sha256=hashlib.sha256(a.read_bytes()).hexdigest(),size=a.stat().st_size)))
else:
    sys.argv=['finalize_mx_evidence.py','{digest}']
    exec(base64.b64decode('{encoded_code}'))
'''
encoded=base64.b64encode(code.encode()).decode()
remote="python3 -c 'import base64;exec(base64.b64decode(\""+encoded+"\"))'"
argv=['C:/Windows/System32/OpenSSH/ssh.exe','-n','-o','BatchMode=yes','-o','ConnectionAttempts=1',
      '-o','ConnectTimeout=8','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','mx',remote]
p=subprocess.run(argv,text=True,capture_output=True,timeout=45)
(HERE/f'finalize_attempt_{attempt}.command.json').write_text(json.dumps(dict(argv=argv,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
if p.returncode:
    print(p.stderr);raise SystemExit(p.returncode)
receipt=json.loads(p.stdout)
(HERE/'MX_ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
