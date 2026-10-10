"""Extract this preflight packet without overwrites, check protected identities, seal evidence."""
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path('/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OPACITY_FIX_R21_P4_MX')
PACKET = ROOT/'preflight_payload.tar'
STAMP = '20261009T150342Z'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

assert sha(PACKET)==sys.argv[1], 'Transferred evidence packet hash mismatch'
with tarfile.open(PACKET,'r:') as tar:
    for item in tar.getmembers():
        path=ROOT/item.name
        assert item.isfile() and not item.issym() and not item.islnk()
        assert ROOT.resolve() in path.resolve().parents and not path.exists()
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as out:
            out.write(tar.extractfile(item).read())
audit=ROOT/'preflight'/STAMP
with (audit/'FINALIZE_COMMAND.json').open('x') as f:
    json.dump(dict(executable=sys.executable,argv=sys.argv,
                   source='preflight/'+STAMP+'/finalize_mx_evidence.py',
                   packet_sha256=sys.argv[1]),f,indent=2)
status=json.loads((audit/'RUN_STATUS.json').read_text())
assert status['status']=='BLOCKED_CANDIDATE_NOT_AVAILABLE'
assert all(s['status']=='NOT_EXECUTED' for s in status['stages'])
post=[]
probe=json.loads((audit/'MX_PREFLIGHT.json').read_text())
for p in probe['paths']:
    if p.get('sha256'):
        actual=sha(Path(p['path']))
        post.append(dict(path=p['path'],before=p['sha256'],after=actual,unchanged=actual==p['sha256']))
assert all(p['unchanged'] for p in post)
with (audit/'PROTECTED_IDENTITY_AFTER.json').open('x') as f:
    json.dump(dict(status='PASS_READ_ONLY_IDENTITY_PRESERVED',files=post),f,indent=2)
with (ROOT/'RUN_STATUS.json').open('x') as f:
    json.dump(status,f,indent=2);f.write('\n')
with (audit/'finalize_mx.log').open('x') as f:
    f.write('Evidence packet SHA256 verified. No existing evidence overwritten.\n'
            'Frozen and R20 audited file hashes unchanged. P4/P5 NOT_EXECUTED.\n'
            'SHA256SUMS and r21_p4_return.tar.gz generated from all evidence below this root.\n')
excluded={'SHA256SUMS','r21_p4_return.tar.gz','r21_p4_return.tar.gz.sha256'}
files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in excluded)
assert not any(p.is_symlink() for p in files)
with (ROOT/'SHA256SUMS').open('x') as f:
    for p in files:f.write(sha(p)+'  '+p.relative_to(ROOT).as_posix()+'\n')
files.append(ROOT/'SHA256SUMS')
archive=ROOT/'r21_p4_return.tar.gz'
with archive.open('xb') as raw:
    with tarfile.open(fileobj=raw,mode='w:gz') as tar:
        for p in files:tar.add(p,arcname=ROOT.name+'/'+p.relative_to(ROOT).as_posix(),recursive=False)
digest=sha(archive)
with (ROOT/'r21_p4_return.tar.gz.sha256').open('x') as f:
    f.write(digest+'  r21_p4_return.tar.gz\n')
print(json.dumps(dict(status=status['status'],archive=str(archive),sha256=digest,
                     size=archive.stat().st_size,evidence_files=len(files),protected_identities_unchanged=True),indent=2))
