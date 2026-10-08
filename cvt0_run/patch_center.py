p = "cvt0_run.py"
s = open(p, encoding="utf-8").read()
old = '    center = cam.camera_center.detach().float().cpu().numpy()'
new = '    center = cam.camera_center.detach().float().cpu().numpy().tolist()'
assert old in s, "center line not found"
s = s.replace(old, new)
# also add a json default safety net on the results dump
old2 = '    with open(os.path.join(args.out, "CVT0_RESULTS.json"), "w") as f:\n        json.dump(results, f, indent=2)'
new2 = '    with open(os.path.join(args.out, "CVT0_RESULTS.json"), "w") as f:\n        json.dump(results, f, indent=2, default=_json_default)'
assert old2 in s, "dump line not found"
s = s.replace(old2, new2)
# add _json_default helper if missing
if "def _json_default(" not in s:
    anchor = "def spearman(x, y):"
    helper = ('def _json_default(o):\n'
              '    import numpy as _np\n'
              '    if isinstance(o, _np.ndarray):\n'
              '        return o.tolist()\n'
              '    if isinstance(o, (_np.floating, _np.integer)):\n'
              '        return o.item()\n'
              '    return str(o)\n\n\n')
    s = s.replace(anchor, helper + anchor, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched center->list + json default safety net")
