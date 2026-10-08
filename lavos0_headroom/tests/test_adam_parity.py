"""Ground-truth test: LAVOS0 settlement == real torch.optim.Adam A_s.

The deepest correctness claim in the protocol. A stateful dense
torch.optim.Adam(foreach=False) stepping with all-zero gradients must equal the
LAVOS0 RowSettler replay primitive BITWISE, and the full defer/settle schedule
must reproduce the dense lane bitwise.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lavos0 import adam_ref                      # noqa: E402


def test_settler_matches_dense_zero_grad():
    r = adam_ref.verify_settlement_parity(feat=5, n_rows=32, n_steps=20, seed=0)
    assert r["bitwise"], r
    assert r["theta_maxabs"] == 0.0
    assert r["m_maxabs"] == 0.0
    assert r["v_maxabs"] == 0.0


def test_settler_deterministic():
    r1 = adam_ref.verify_settlement_parity(feat=4, n_rows=16, n_steps=15,
                                           seed=3)
    r2 = adam_ref.verify_settlement_parity(feat=4, n_rows=16, n_steps=15,
                                           seed=3)
    assert r1 == r2 and r1["bitwise"]


def test_settler_handles_no_consumers():
    # every row zero-grad every step: deferral must still settle to dense.
    r = adam_ref.verify_settlement_parity(feat=3, n_rows=8, n_steps=10, seed=1)
    assert r["bitwise"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL", len(fns), "adam_parity tests passed")
