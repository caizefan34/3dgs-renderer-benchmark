"""Download the MX return archive and verify every sealed raw-evidence hash."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
ROOT='/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OPACITY_FIX_R21_P4_MX'
download=HERE/'download';download.mkdir(exist_ok=True)
names=['r21_p4_return.tar.gz','r21_p4_return.tar.gz.sha256']
for name in names:assert not (download/name).exists()
remote_code="import base64,json;from pathlib import Path;r=Path('"+ROOT+"');print(json.dumps({n:base64.b64encode((r/n).read_bytes()).decode() for n in "+repr(names)+"}))"
encoded=base64.b64encode(remote_code.encode()).decode()
remote="python3 -c 'import base64;exec(base64.b64decode(\""+encoded+"\"))'"
argv=['C:/Windows/System32/OpenSSH/ssh.exe','-n','-o','BatchMode=yes','-o','ConnectionAttempts=1',
      '-o','ConnectTimeout=8','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',
      'mx',remote]
p=subprocess.run(argv,text=True,capture_output=True,timeout=45)
(HERE/'download_ssh.command.json').write_text(json.dumps(dict(argv=argv,exit_code=p.returncode,stdout_file='download_ssh.stdout.log',stderr=p.stderr),indent=2)+'\n')
(HERE/'download_ssh.stdout.log').write_text(p.stdout)
assert p.returncode==0,p.stderr
payload=json.loads(p.stdout)
for name in names:
    with (download/name).open('xb') as f:f.write(base64.b64decode(payload[name],validate=True))
archive=download/'r21_p4_return.tar.gz'
actual=hashlib.sha256(archive.read_bytes()).hexdigest()
receipt=json.loads((HERE/'MX_ARCHIVE_RECEIPT.json').read_text())
assert actual==receipt['sha256']==(download/'r21_p4_return.tar.gz.sha256').read_text().split()[0]
with tarfile.open(archive,'r:gz') as tar:
    prefix=Path(ROOT).name+'/'
    checks=tar.extractfile(prefix+'SHA256SUMS').read().decode().splitlines()
    for line in checks:
        expected,name=line.split('  ',1)
        assert hashlib.sha256(tar.extractfile(prefix+name).read()).hexdigest()==expected,name
    status=json.loads(tar.extractfile(prefix+'RUN_STATUS.json').read())
    assert status['status']=='BLOCKED_CANDIDATE_NOT_AVAILABLE'
    assert all(s['status']=='NOT_EXECUTED' for s in status['stages'])
    identities=json.loads(tar.extractfile(prefix+'preflight/'+HERE.name+'/PROTECTED_IDENTITY_AFTER.json').read())
    assert all(record['unchanged'] for record in identities['files'])
    (download/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    (download/'SHA256SUMS').write_text('\n'.join(checks)+'\n')
result=dict(status='PASS_ARCHIVE_INTEGRITY_ONLY',archive=str(archive),sha256=actual,
            evidence_hashes_verified=len(checks),execution_status=status['status'])
(HERE/'ARCHIVE_INTEGRITY_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
