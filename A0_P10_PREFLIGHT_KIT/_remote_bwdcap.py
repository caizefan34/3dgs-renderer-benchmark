import os
for p in ("/root/epic05_frozen/higs_c0_final30k_worktree/backward_capture.py",
          "/root/epic05_frozen/higs_c0_final30k_worktree/bootstrap.sh"):
    if os.path.exists(p):
        print(f"===== {p} =====")
        print(open(p).read()[:4000])
        print()
# also find where _native_backward captures conics/tile_offsets/flatten_ids/render_alphas/last_ids
src = open("/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/functional/gaussian_inference.py").read().splitlines()
print("===== _native_backward: state capture (conics/tile_offsets/flatten_ids/render_alphas/last_ids) =====")
for i,l in enumerate(src,1):
    if any(k in l for k in ("tile_offsets","flatten_ids","render_alphas","last_ids","conics","colors_eval","opacities","capture_backward","backward_state","_capture")) and 2280 < i < 2700:
        print(f"{i:5d} {l.rstrip()[:140]}")
