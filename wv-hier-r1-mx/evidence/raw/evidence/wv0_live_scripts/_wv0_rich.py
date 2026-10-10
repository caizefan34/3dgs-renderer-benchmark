import importlib.util as u
for mod in ["rich","gsplat","torch"]:
    spec=u.find_spec(mod)
    print(mod, spec.origin if spec else "MISSING")
try:
    import gsplat.cuda._backend as b
    print("backend import OK")
except Exception as e:
    print("backend ERR", repr(e)[:200])
