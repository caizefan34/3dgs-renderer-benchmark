"""Static fingerprints; no scientific imports."""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL_FILES=('METHOD.md','SOLVER_SPEC.md','ADAPTER_CONTRACT.md','PROTOCOL.md',
                'GATES.md','config/prereg.json','config/mx_target.json')
def digest_files(paths):
    h=hashlib.sha256()
    for p in sorted(paths,key=lambda p:p.relative_to(ROOT).as_posix()):
        h.update(p.relative_to(ROOT).as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
    return h.hexdigest()
def protocol_digest():
    return digest_files([ROOT/p for p in PROTOCOL_FILES])
def implementation_digest():
    return digest_files([p for p in ROOT.rglob('*') if p.is_file() and
                         p.suffix in ('.py','.cu','.cuh','.cpp','.h','.hpp')])
def reference_digest():
    return digest_files([ROOT/'scripts'/p for p in ('mss_reference.py','run_e0.py','mx_guard.py')])
