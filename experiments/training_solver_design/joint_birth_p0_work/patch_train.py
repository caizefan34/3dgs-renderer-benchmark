from pathlib import Path

p = Path("/root/sadgs_joint_birth_p0_20260928/train.py")
s = p.read_text()
key = "from scene import Scene, GaussianModel\n"
add = key + '''if os.getenv("SADGS_JB_CAPTURE_DIR"):
    from jb_p0_capture import install as _install_jb_p0_capture
    _install_jb_p0_capture(GaussianModel)
'''
assert s.count(key) == 1
p.write_text(s.replace(key, add))
