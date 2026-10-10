"""Capture exact read-only MX audit commands and their unabridged outputs."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
for name in ("commands", "logs", "raw", "gpu_snapshots"):
    (ROOT / name).mkdir(exist_ok=True)
script = Path(sys.argv[1])
label = script.stem
body = script.read_bytes()
target = ROOT / "commands" / script.name
if target.resolve() != script.resolve():
    target.write_bytes(body)
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
t0 = time.monotonic()
proc = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "mx", "bash -s"],
    input=body, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
(ROOT / "logs" / (label + ".stdout.log")).write_bytes(proc.stdout)
(ROOT / "logs" / (label + ".stderr.log")).write_bytes(proc.stderr)
(ROOT / "logs" / (label + ".meta.json")).write_text(json.dumps({
    "started_utc": started, "elapsed_wall_seconds": time.monotonic() - t0,
    "returncode": proc.returncode, "command": "ssh -o BatchMode=yes -o ConnectTimeout=15 mx bash -s",
    "stdin_file": "commands/" + script.name,
}, indent=2), encoding="utf-8")
print(proc.stdout.decode("utf-8", errors="replace"))
print(proc.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
sys.exit(proc.returncode)
