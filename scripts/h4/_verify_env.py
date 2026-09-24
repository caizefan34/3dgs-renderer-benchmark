import sys

sys.path.insert(0, "/tmp")
import h2_bwd_0_structural as h

print("h2 imports OK:", h.TILE_SIZE, h.SH_DEGREE)
import numpy as np

print("numpy OK:", np.__version__)
