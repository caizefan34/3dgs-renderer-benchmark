"""Standard-library fingerprints; no scientific computation or environment imports."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL_FILES=("METHOD.md","NUMERICS.md","IMPLEMENTATION.md","ADAPTER_CONTRACT.md","PROTOCOL.md",
                "GATES.md","config/prereg.json","config/mx_target.json")

def digest_files(paths):
    h=hashlib.sha256()
    for path in sorted(paths,key=lambda p:p.relative_to(ROOT).as_posix()):
        h.update(path.relative_to(ROOT).as_posix().encode("utf-8")+b"\0")
        h.update(path.read_bytes()+b"\0")
    return h.hexdigest()

def protocol_digest():
    return digest_files([ROOT/name for name in PROTOCOL_FILES])

def implementation_digest():
    return digest_files([p for p in ROOT.rglob("*") if p.is_file() and
                         p.suffix in (".py",".cu",".cuh",".cpp",".h",".hpp")])

def selected_config_digest():
    """SHA256 of the frozen config/selected_config.json. Computed at runtime
    so drivers do not hardcode a hash of a file that itself embeds the
    implementation digest (avoids a driver<->config hash cycle)."""
    import hashlib
    p = ROOT / "config" / "selected_config.json"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reference_digest():
    return digest_files([ROOT/'scripts'/name for name in
                         ('bes_reference.py','run_e0.py','mx_guard.py')])
