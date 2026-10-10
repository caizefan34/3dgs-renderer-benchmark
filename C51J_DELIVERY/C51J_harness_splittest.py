"""
Local unit test for C51J_harness SplitIsolator (no GPU needed).
Validates: train/val/heldout are disjoint and cover all cameras; assert_train_only raises on val/heldout.
Run: python C51J_harness_splittest.py
"""
import sys, importlib.util
spec = importlib.util.spec_from_file_location("h", r"C:\Users\36570\3dgs-renderer-benchmark\C51J_DELIVERY\C51J_harness.py")
# We can't exec the whole module (it imports torch at gate time via functions only);
# SplitIsolator only needs numpy, so exec just the class defs by loading source and
# stubbing the parts that need torch. Simplest: copy the class logic test via import
# with a guard. The module's top-level has no torch import (all torch imports are
# inside functions), so a direct import should work with numpy present.
import numpy as np  # noqa

# Extract and exec only the SplitIsolator class to avoid any torch dependency.
src = open(r"C:\Users\36570\3dgs-renderer-benchmark\C51J_DELIVERY\C51J_harness.py", encoding="utf-8").read()
start = src.index("class SplitIsolator:")
end = src.index("# ---", start)
ns = {"np": np}
exec(src[start:end], ns)
SplitIsolator = ns["SplitIsolator"]

def test_scene(n_cams, scene):
    s = SplitIsolator(n_cams, scene, seed=0)
    train, val, held = set(s.train), s.val, s.heldout
    # disjoint
    assert train.isdisjoint(val), f"{scene}: train∩val={train&val}"
    assert train.isdisjoint(held), f"{scene}: train∩heldout={train&held}"
    assert val.isdisjoint(held), f"{scene}: val∩heldout={val&held}"
    # exhaustive
    union = train | val | held
    assert union == set(range(n_cams)), f"{scene}: missing {set(range(n_cams))-union}, extra {union-set(range(n_cams))}"
    # no leak on train
    for c in list(train)[:5]:
        s.assert_train_only(c, "train")  # must not raise
    # leak raises on val
    leak_val = leak_held = False
    if val:
        v = next(iter(val))
        try: s.assert_train_only(v, "test")
        except RuntimeError: leak_val = True
    if held:
        h = next(iter(held))
        try: s.assert_train_only(h, "test")
        except RuntimeError: leak_held = True
    return {"scene": scene, "n": n_cams, "train": len(train), "val": len(val),
            "heldout": len(held), "leak_val_raises": leak_val, "leak_held_raises": leak_held}

results = [test_scene(n, s) for n, s in [(311, "room"), (307, "bicycle"), (300, "garden"), (1, "tiny"), (8, "eight")]]
ok = all(r["leak_val_raises"] or r["val"] == 0 for r in results) and all(r["leak_held_raises"] or r["heldout"] == 0 for r in results)
for r in results:
    print(r)
print("SPLIT_ISOLATOR_TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
