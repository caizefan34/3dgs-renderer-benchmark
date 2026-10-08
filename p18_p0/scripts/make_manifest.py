#!/usr/bin/env python3
import hashlib, json, os, time
from pathlib import Path
ROOT = Path(r"C:\Users\36570\3dgs-renderer-benchmark\p18_p0")
ART = ROOT / "artifacts"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def entry(p):
    p = Path(p)
    return {"path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": p.stat().st_size, "sha256": sha(p), "sha256_short": sha(p)[:16]}

files = []
for d, _, fs in os.walk(ROOT):
    dp = Path(d)
    if dp == ART / "traces":
        continue  # NPZ traces listed separately
    for fn in sorted(fs):
        fp = dp / fn
        if fp.suffix in (".pyc",) or "__pycache__" in str(fp):
            continue
        files.append(entry(fp))

manifest = {
    "phase": "P18-P0",
    "artifact": "MANIFEST_P18",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "verdict": "P18_P0_HARD_DROP",
    "root": str(ROOT).replace("\\", "/"),
    "frozen_baseline_hashes": {
        "trainer": "675b026711b1938b28949c261c4f8e9e04d4a0a2846e2226c11d086b7999d28f50c5",
        "c0_ext": "d26cccf45868386e214650958838c6b6d654f1a55e6003c6a8007409554e637e",
        "core": "935ea6c099355e824d62b2a38b3d57580e578c140e47c48f533a0c6b1c880c7e",
        "fused_ssim": "dd6a4c79a0735d4c43a13b289d08e234270528b29a2b207827d0f0728e250ebb",
        "ckpt_room": "17c1f590110a803f9e594d5671179d4e9f19c4c822c9a57638490219062a0e80",
        "ckpt_bicycle": "0c36f5c7f42997174283d4e378b239875412509f23865e4a455605a761702995",
    },
    "files": files,
    "npz_traces": [
        entry(ART/"traces"/"P18_TRACE_room.npz"),
        entry(ART/"traces"/"P18_TRACE_bicycle.npz"),
    ],
    "note": "P18_FORWARD_FIDELITY.json / P18_BACKWARD_FIDELITY.json / P18_INTEGRATED_TIMING.json / P18_DIRECT_REPEAT_NOISE.json are intentionally absent (phases not run; HARD DROP fired at the evaluator stage).",
}
out = ART / "MANIFEST_P18.json"
with open(out, "w") as f:
    json.dump(manifest, f, indent=2, allow_nan=False)
print("wrote", out)
for e in manifest["files"]:
    print(f"  {e['sha256_short']}  {e['size_bytes']:>10}  {e['path']}")
