import gsplat, os
print("gsplat path:", list(gsplat.__path__))
# Check if it's an editable install
import site
site_pkgs = site.getsitepackages()
user_site = site.getusersitepackages()
print("user_site:", user_site)
# Check for egg-link or pth files
for d in [user_site] + site_pkgs:
    if os.path.isdir(d):
        for f in os.listdir(d):
            if 'gsplat' in f.lower() and (f.endswith('.egg-link') or f.endswith('.pth') or f.endswith('.egg-info')):
                print(f"  Found: {os.path.join(d, f)}")
                if f.endswith('.egg-link'):
                    with open(os.path.join(d, f)) as fh:
                        print(f"    -> {fh.read().strip()}")
# Check where csrc.so actually is
gsplat_dir = list(gsplat.__path__)[0]
print("gsplat_dir:", gsplat_dir)
print("csrc.so exists:", os.path.exists(os.path.join(gsplat_dir, "csrc.so")))
print("csrc.so in /tmp:", os.path.exists("/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/csrc.so"))
