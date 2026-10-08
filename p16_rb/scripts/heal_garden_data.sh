#!/bin/bash
# P16-RB garden data heal (operator-approved 2026-09-28).
# Installs the official MipNeRF360 garden sparse (range-fetched from the gresearch
# 360_v2.zip, CRC/manifest-verified) into the shared dataset, regenerates points3D.ply
# via the method loader code path, and restages the common dataset (garden only).
# Corrupt files are copied to a backup dir first -- nothing is deleted.
set -u
D=/root/epic05_p16rb
REPO=/root/3dgs-renderer-benchmark
SP_REPO=$REPO/data/datasets/mipnerf360/garden/sparse/0
SP_COM=$D/common_data/mipnerf360/garden/sparse/0
OFF=$D/garden_repair/official/sparse/0
MAN=$D/garden_repair/official/MANIFEST_garden_official.json
BK=$D/backups/garden_corrupt_20260928
PY=/root/miniforge3/envs/p16_fastergs/bin/python
BICYCLE_BIN=$REPO/data/datasets/mipnerf360/bicycle/sparse/0/points3D.bin
BICYCLE_PLY=$REPO/data/datasets/mipnerf360/bicycle/sparse/0/points3D.ply
LOG=$D/logs/garden_heal.log
: > "$LOG"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "HEAL_START"
mkdir -p "$BK/repo_sparse0" "$BK/common_sparse0"

log "pre-heal repo listing:"
ls -la "$SP_REPO" | tee -a "$LOG"
log "pre-heal common listing:"
ls -la "$SP_COM" | tee -a "$LOG"

# H0: verify the fetched official files against the fetch manifest
"$PY" - "$MAN" "$OFF" <<'PYEOF' >> "$LOG" 2>&1 || { log "H0 FAILED"; exit 2; }
import hashlib, json, sys
from pathlib import Path
man = json.loads(Path(sys.argv[1]).read_text())
off = Path(sys.argv[2])

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

for m in man["members"]:
    if m["name"].startswith("garden/sparse/"):
        p = off / Path(m["name"]).name
        got = sha256(p)
        assert got == m["sha256"], (m["name"], got)
        print("official-ok", p.name, m["bytes"], got[:16])
PYEOF
log "H0 official sparse verified against fetch manifest"

# H1: back up the corrupt files (copy, never delete)
for f in cameras.bin images.bin points3D.bin points3D.ply; do
  [ -f "$SP_REPO/$f" ] && cp -p "$SP_REPO/$f" "$BK/repo_sparse0/$f"
  [ -f "$SP_COM/$f" ] && cp -p "$SP_COM/$f" "$BK/common_sparse0/$f"
done
( cd "$BK" && sha256sum repo_sparse0/* common_sparse0/* > SHA256SUMS.txt )
log "H1 corrupt files backed up: $BK (with SHA256SUMS.txt)"

# H2: install the official sparse into the repo dataset
for f in cameras.bin images.bin points3D.bin; do
  cp -f "$OFF/$f" "$SP_REPO/$f"
done
log "H2 official sparse installed into repo garden"

# H3: regenerate points3D.ply via the FastGS loader code path.
# Validate the generator against the healthy bicycle ply first.
CHK=$D/garden_repair/plycheck
mkdir -p "$CHK"
"$PY" $D/scripts/regen_ply_from_bin.py --repo-root $D/src/FastGS \
  --bin "$BICYCLE_BIN" --out "$CHK/bicycle_points3D.ply" >> "$LOG" 2>&1
if cmp -s "$CHK/bicycle_points3D.ply" "$BICYCLE_PLY"; then
  PLY_OK=byte_identical
  log "H3 ply-generator validation: bicycle regen BYTE-IDENTICAL to shipped ply"
else
  if "$PY" - "$CHK/bicycle_points3D.ply" "$BICYCLE_PLY" <<'PYEOF' >> "$LOG" 2>&1
import sys
import numpy as np
from plyfile import PlyData
a = PlyData.read(sys.argv[1])["vertex"]
b = PlyData.read(sys.argv[2])["vertex"]
assert a.count == b.count, (a.count, b.count)
for f in ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"):
    assert np.array_equal(np.asarray(a[f]), np.asarray(b[f])), f
print("bicycle ply array-identical", a.count)
PYEOF
  then
    PLY_OK=array_identical
  else
    PLY_OK=FAIL
  fi
  log "H3 ply-generator validation: $PLY_OK"
fi
[ "$PLY_OK" = FAIL ] && { log "H3 FAILED"; exit 3; }

"$PY" $D/scripts/regen_ply_from_bin.py --repo-root $D/src/FastGS \
  --bin "$SP_REPO/points3D.bin" --out "$SP_REPO/points3D.ply" >> "$LOG" 2>&1
[ -s "$SP_REPO/points3D.ply" ] || { log "H3 garden ply regen FAILED"; exit 3; }
log "H3 repo garden points3D.ply regenerated"
ls -la "$SP_REPO" | tee -a "$LOG"

# H3b: load-consistency -- fetchPly arrays must equal read_points3D_binary arrays
"$PY" - "$SP_REPO" <<'PYEOF' >> "$LOG" 2>&1 || { log "H3b FAILED"; exit 3; }
import sys
sys.path.insert(0, "/root/epic05_p16rb/src/FastGS")
import numpy as np
from scene.dataset_readers import read_points3D_binary, fetchPly
sp = sys.argv[1]
xyz, rgb, _ = read_points3D_binary(sp + "/points3D.bin")
p = fetchPly(sp + "/points3D.ply")
assert len(xyz) == len(p.points), (len(xyz), len(p.points))
assert np.array_equal(xyz.astype("f4"), p.points.astype("f4"))
assert np.array_equal(np.asarray(rgb).astype("u1"), np.asarray(p.colors * 255, dtype="u1"))
z = xyz[:, 2]
print("H3b load-consistent n_points=%d z[min %.3f max %.3f std %.3f]" % (len(xyz), z.min(), z.max(), z.std()))
assert z.std() > 1.0, "z flat -- still corrupt"
PYEOF
log "H3b ply/bin load-consistency OK"

# H4: pycolmap strict load of the healed repo sparse
"$PY" - "$SP_REPO" <<'PYEOF' >> "$LOG" 2>&1 || { log "H4 pycolmap load FAILED"; exit 4; }
import sys
import pycolmap
rec = pycolmap.Reconstruction(sys.argv[1])
n_img, n_pts = len(rec.images), len(rec.points3D)
cam = list(rec.cameras.values())[0]
print("pycolmap: %d images, %d points3D, camera model %s %dx%d" % (n_img, n_pts, cam.model, cam.width, cam.height))
assert n_img == 185 and n_pts == 138766, (n_img, n_pts)
PYEOF
log "H4 pycolmap strict load OK"

# H5: garden-only common restage (rebuilds PNGs, images.bin, copies healed sparse;
# merges the garden entry into P16_COMMON_DATA_BUILD.json)
cd $D
"$PY" $D/scripts/p16_build_common_data.py garden >> "$LOG" 2>&1 || { log "H5 restage FAILED"; exit 5; }
log "H5 common restage done"

# H6: final verification of the healed trees
"$PY" - "$SP_REPO" "$SP_COM" "$D" <<'PYEOF' >> "$LOG" 2>&1 || { log "H6 verification FAILED"; exit 6; }
import hashlib, json, sys
from pathlib import Path

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

sp, com, D = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
want = {
    "cameras.bin": "8472cc1a821db10de24155a120abfacec9268c39b562b924f8993ca0e0f299e4",
    "images.bin": "9490fc63d509ab5bcff3d85771402dd6c623809dbf0fa250a05a29bb3cea8c41",
    "points3D.bin": "be52e343ad059bbfce00e2b22d397de75dad2d6f1ac7cf359b1b926c2b0482c8",
}
for f, w in want.items():
    g = sha256(sp / f)
    assert g == w, (f, g)
    print("repo", f, "OK", g[:16])
assert sha256(com / "points3D.bin") == want["points3D.bin"], "common points3D.bin mismatch"
assert sha256(com / "points3D.ply") == sha256(sp / "points3D.ply"), "common ply != repo ply"
assert (com / "images.bin").stat().st_size == 47011789, (com / "images.bin").stat().st_size
print("common sparse OK (images.bin 47011789B)")
rep = json.loads((D / "common_data/mipnerf360/P16_COMMON_DATA_BUILD.json").read_text())
assert set(rep) >= {"room", "bicycle", "garden"}, sorted(rep)
g = rep["garden"]
assert g["n_images_written"] == 185 and not g["names_missing"], g
assert g["points3D_bin_sha256"] == want["points3D.bin"], g["points3D_bin_sha256"]
print("build report OK: garden %d images, points3D %s" % (g["n_images_written"], g["points3D_bin_sha256"][:16]))
PYEOF
log "H6 verification OK"

log "HEAL_DONE"
