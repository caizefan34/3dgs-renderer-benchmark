import glob, os, torch
# find gsplat csrc.so under each env and its torch version
home=os.path.expanduser("~")
envs=["strong_native","nerficg","anysplat","chorus"]
for e in envs:
    tv=torch.__version__
    # env's own site-packages gsplat
    base=f"/home/liaoyuanjun/miniforge3/envs/{e}/lib"
    so=glob.glob(base+"/python*/site-packages/gsplat/csrc.so")
    us=f"{home}/.local/lib/python3.10/site-packages/gsplat/csrc.so"
    print(e,"torch",tv,"env_gsplat_so",so, "exists",os.path.exists(so[0]) if so else False)
    print("   usersite_so exists", os.path.exists(us))
