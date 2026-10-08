from pathlib import Path

p = Path("/root/sadgs_joint_birth_p0_20260928/train.py")
s = p.read_text()
key1 = "    viewpoint_stack = scene.getTrainCameras().copy()\n    viewpoint_indices = list(range(len(viewpoint_stack)))\n"
block1 = (
    '    if os.getenv("SADGS_JB_CAPTURE_DIR"):\n'
    '        ordered_cams = sorted(scene.getTrainCameras(), key=lambda cam: cam.image_name)\n'
    '        selected = [round(k * (len(ordered_cams) - 1) / 11) for k in range(12)]\n'
    '        gaussians._jb_a_camera_names = [ordered_cams[selected[k]].image_name for k in range(0, 12, 2)]\n'
    '        gaussians._jb_validation_camera = ordered_cams[selected[0]]\n'
    '        gaussians._jb_render_context = (pipe, background, opt.mult)\n'
    '        gaussians._jb_training_options = dict(vars(opt))\n'
)
assert s.count(key1) == 1
if "gaussians._jb_a_camera_names" not in s:
    s = s.replace(key1, key1 + block1)
key2 = "                    gaussians.densify_and_prune_structgs(\n"
block2 = (
    '                    if hasattr(gaussians, "_jb_training_options"):\n'
    '                        gaussians._jb_context = {\n'
    '                            "iteration": int(iteration),\n'
    '                            "current_batch_last_camera": viewpoint_cam.image_name,\n'
    '                            "remaining_camera_stack": [cam.image_name for cam in viewpoint_stack],\n'
    '                            "optimizer_lrs": {g.get("name", str(i)): float(g["lr"]) for i, g in enumerate(gaussians.optimizer.param_groups)},\n'
    '                        }\n'
)
assert s.count(key2) == 1
if "remaining_camera_stack" not in s:
    s = s.replace(key2, block2 + key2)
p.write_text(s)
