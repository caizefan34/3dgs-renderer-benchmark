import sys, hashlib
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()
live_c0 = sha(C.C0_SO); live_core = sha(C.CORE_SO)
prereg_c0 = "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11"
prereg_core = "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"
print("live_c0  ", live_c0, len(live_c0))
print("prereg_c0", prereg_c0, len(prereg_c0))
print("c0 equal:", live_c0 == prereg_c0)
print("live_core  ", live_core, len(live_core))
print("prereg_core", prereg_core, len(prereg_core))
print("core equal:", live_core == prereg_core)
# find first diff
for i,(a,b) in enumerate(zip(live_c0, prereg_c0)):
    if a!=b:
        print("first c0 diff at", i, a, b); break
