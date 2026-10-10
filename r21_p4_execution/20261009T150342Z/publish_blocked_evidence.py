"""Publish only audit evidence to a new output root; do not transfer/deploy candidate runtime."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
ROOT='/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OPACITY_FIX_R21_P4_MX'
SSH='C:/Windows/System32/OpenSSH/ssh.exe'
SCP='C:/Windows/System32/OpenSSH/scp.exe'
OPTIONS=['-o','BatchMode=yes','-o','ConnectionAttempts=1','-o','ConnectTimeout=8',
         '-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2']

def run(name,argv):
    p=subprocess.run(argv,text=True,capture_output=True,encoding='utf-8',errors='replace',timeout=90)
    (HERE/(name+'.command.json')).write_text(json.dumps(dict(argv=argv,exit_code=p.returncode,
                    stdout=p.stdout,stderr=p.stderr),indent=2)+'\n',encoding='utf-8')
    (HERE/(name+'.stdout.log')).write_text(p.stdout,encoding='utf-8')
    (HERE/(name+'.stderr.log')).write_text(p.stderr,encoding='utf-8')
    if p.returncode:raise RuntimeError(name+': '+p.stderr)
    return p.stdout

def remote(code,args=''):
    encoded=base64.b64encode(code).decode()
    return "python3 -c 'import base64;exec(base64.b64decode(\""+encoded+"\"))'"+args

status=json.loads((HERE/'RUN_STATUS.json').read_text())
assert status['status']=='BLOCKED_CANDIDATE_NOT_AVAILABLE'
create=("from pathlib import Path; p=Path('"+ROOT+"'); "
        "assert p.parent.resolve()==Path('/mnt/storage_pool/liaoyuanjun'); "
        "p.mkdir(exist_ok=False); print('CREATED_EMPTY_EVIDENCE_ROOT '+str(p))").encode()
run('mx_create_output_root',[SSH,'-n',*OPTIONS,'mx',remote(create)])
packet=HERE/'preflight_payload.tar'
inputs=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
with tarfile.open(packet,'x:') as tar:
    for p in inputs:tar.add(p,arcname='preflight/'+HERE.name+'/'+p.relative_to(HERE).as_posix(),recursive=False)
digest=hashlib.sha256(packet.read_bytes()).hexdigest()
run('mx_upload_preflight',[SCP,*OPTIONS,str(packet),'mx:'+ROOT+'/preflight_payload.tar'])
result=run('mx_finalize_evidence',[SSH,'-n',*OPTIONS,'mx',remote((HERE/'finalize_mx_evidence.py').read_bytes(),' '+digest)])
receipt=json.loads(result)
(HERE/'MX_ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
download=HERE/'download';download.mkdir(exist_ok=False)
for name in ['r21_p4_return.tar.gz','r21_p4_return.tar.gz.sha256','SHA256SUMS','RUN_STATUS.json']:
    run('download_'+name.replace('.','_'),[SCP,*OPTIONS,'mx:'+ROOT+'/'+name,str(download/name)])
archive=download/'r21_p4_return.tar.gz'
actual=hashlib.sha256(archive.read_bytes()).hexdigest()
assert actual==receipt['sha256']==(download/'r21_p4_return.tar.gz.sha256').read_text().split()[0]
with tarfile.open(archive,'r:gz') as tar:
    prefix=Path(ROOT).name+'/'
    checks=tar.extractfile(prefix+'SHA256SUMS').read().decode().splitlines()
    verified=0
    for line in checks:
        expected,name=line.split('  ',1)
        assert hashlib.sha256(tar.extractfile(prefix+name).read()).hexdigest()==expected,name
        verified+=1
    archived_status=json.loads(tar.extractfile(prefix+'RUN_STATUS.json').read())
    assert archived_status['status']=='BLOCKED_CANDIDATE_NOT_AVAILABLE'
    assert all(s['status']=='NOT_EXECUTED' for s in archived_status['stages'])
verification=dict(status='PASS_ARCHIVE_INTEGRITY_ONLY',archive=str(archive),sha256=actual,
                  evidence_hashes_verified=verified,execution_status=archived_status['status'])
(HERE/'ARCHIVE_INTEGRITY_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
