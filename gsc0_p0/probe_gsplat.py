import sys, os, glob, traceback
REPO="/root/3dgs-renderer-benchmark"
SRC=os.path.join(REPO,"src")
REF=os.path.join(REPO,"baseline","reference_v1")
PHASE7=os.path.join(REPO,"scripts","epic05","phase7")

# Candidate gsplat source trees
trees = glob.glob("/root/epic05_frozen/gsplat-*-v*/gsplat") + \
        glob.glob("/root/epic05_p6/*/gsplat-p6-v153/gsplat") + \
        ["/root/epic05_frozen/gsplat-b1-clean-v153/gsplat"]
trees = [t for t in trees if os.path.isdir(t)]

def has_wrapper(tree):
    return os.path.exists(os.path.join(tree,"cuda","_wrapper.py")) or \
           os.path.exists(os.path.join(tree,"cuda","_wrapper.abi3.so"))

print("=== candidate gsplat trees with cuda wrapper ===")
for t in trees:
    print(" ", t, "wrapper=", has_wrapper(t))

print("\n=== try import each tree ===")
for t in trees:
    if not has_wrapper(t):
        continue
    sys.path.insert(0, os.path.dirname(t))  # parent so 'gsplat' package resolves
    for m in list(sys.modules):
        if m=="gsplat" or m.startswith("gsplat."):
            del sys.modules[m]
    try:
        import gsplat
        from gsplat import rasterization
        print("OK import", t, "->", gsplat.__file__, "ver", getattr(gsplat,"__version__","?"))
        break
    except Exception as e:
        print("FAIL", t, "->", repr(e)[:200])
        tb = traceback.format_exc().splitlines()
        for ln in tb[-4:]:
            print("    ", ln)