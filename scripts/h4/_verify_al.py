import inspect
import sys

sys.path.insert(0, "/tmp")

import h3_fwd_0_al as m

print("module OK")
print("has f2_b2:", hasattr(m, "f2_b2"))
print("signature:", inspect.signature(m.f2_b2))
