#!/bin/bash
# D0 Stage 0b: build the traced worktree copy on EPIC-05 and verify integrity.
set -euo pipefail
SRC=/root/epic05_d0/scripts
ART=/root/epic05_d0/artifacts
WT=/root/epic05_d0/c0_worktree_d0
FROZEN=/root/epic05_frozen/higs_c0_final30k_worktree
REL=gsplat/experimental/render/functional/gaussian_inference.py
mkdir -p "$ART"

test -f "$FROZEN/$REL"
sha256sum "$FROZEN/$REL" > "$ART/frozen_renderer_sha_before.txt"
cat "$ART/frozen_renderer_sha_before.txt"

if [ -d "$WT" ]; then
  test "$WT" = "/root/epic05_d0/c0_worktree_d0"
  rm -rf "$WT"
fi
cp -a "$FROZEN" "$WT"

python3 "$SRC/patch_renderer.py" "$WT" "$ART/D0_RENDERER_DIFF.diff"
nhunks=$(grep -c '^@@' "$ART/D0_RENDERER_DIFF.diff")
test "$nhunks" = "5" || { echo "renderer diff hunks=$nhunks (expected 5)"; exit 1; }
echo "renderer diff hunks=5 OK"

sha256sum "$FROZEN/$REL" > "$ART/frozen_renderer_sha_after.txt"
diff "$ART/frozen_renderer_sha_before.txt" "$ART/frozen_renderer_sha_after.txt"
echo "frozen worktree untouched OK"

python3 - "$SRC" "$WT" <<'PY'
import ast, os, sys
src, wt = sys.argv[1], sys.argv[2]
for f in ("d0_trainer.py", "d0_gaussian_model.py", "d0_trace.py"):
    ast.parse(open(os.path.join(src, f)).read())
    print("ast OK", f)
p = os.path.join(wt, "gsplat/experimental/render/functional/gaussian_inference.py")
t = open(p).read()
for k in ("d0_tiles_per_gauss", "d0_depths_vis", "d0_means2d_px", "d0_f9_enabled"):
    n = t.count('"%s"' % k)
    assert n >= 3, (k, n)
    print("d0 key OK", k, "occurrences:", n)
assert "ctx.d0_tiles_per_gauss = tiles_per_gauss.detach()" in t
assert "ctx.d0_means2d_px = captured[0].detach()" in t
assert t.count('"d0_tiles_per_gauss": fwd_meta.get("d0_tiles_per_gauss")') == 1, \
    "R5 dynamic-metadata insertion missing or duplicated"
print("worktree patch verified")
PY

sha256sum "$SRC"/d0_trainer.py "$SRC"/d0_gaussian_model.py "$SRC"/d0_trace.py \
          "$SRC"/patch_renderer.py "$WT/$REL" | tee "$ART/d0_source_hashes.txt"

# Stage 0a: frozen identity re-verification (prereg hashes)
echo "---- frozen identity ----"
sha256sum /root/epic05_p12/publication_trainer_p12.py
sha256sum /root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so
sha256sum /root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so
python3 - <<'PY'
import hashlib, glob, os
want = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"
hits = []
for p in glob.glob("/root/miniforge3/envs/gsplat/lib/python*/site-packages/fused_ssim_cuda*.so"):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    hits.append((p, h, h == want))
for p, h, ok in hits:
    print("fused_ssim_cuda", p, h, "MATCH" if ok else "MISMATCH")
assert any(ok for _, _, ok in hits), "fused-ssim .so hash mismatch"
PY
