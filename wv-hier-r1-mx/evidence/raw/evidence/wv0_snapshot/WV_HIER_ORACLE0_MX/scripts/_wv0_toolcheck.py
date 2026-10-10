import glob, os, subprocess
# nvcc?
for c in ["nvcc","ninja","gcc","g++"]:
    p=subprocess.run(["bash","-lc",f"command -v {c} && {c} --version 2>&1 | head -1"],capture_output=True,text=True)
    print(c, "->", p.stdout.strip().replace("\n"," ")[:120] or "MISSING")
# ninja module in any env
for e in ["anysplat","chorus","nerficg","strong_native"]:
    r=subprocess.run([f"/home/liaoyuanjun/miniforge3/envs/{e}/bin/python","-c","import ninja,sys; print(ninja.BIN_DIR if hasattr(ninja,'BIN_DIR') else ninja.__file__)"],capture_output=True,text=True)
    print(e,"ninja",r.stdout.strip(),r.stderr.strip()[:80])
