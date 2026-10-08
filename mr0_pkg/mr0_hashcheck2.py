import hashlib, glob, os, sys
import gsplat
d = os.path.dirname(gsplat.__file__)
print("GSPAT_DIR", d)
seen = set()
for f in glob.glob(d + "/**/*.so", recursive=True) + glob.glob(d + "/**/*.so.*", recursive=True):
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    print(h[:16], os.path.getsize(f), os.path.relpath(f, d))
    seen.add(h[:16])
print("N_SO", len(seen))
# also site-packages top-level .so
import site
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    for f in glob.glob(sp + "/*.so"):
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        print("SP", h[:16], os.path.basename(f))
