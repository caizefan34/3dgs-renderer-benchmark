"""Deterministic hashing + JSON IO helpers for LAVOS0.

Exactness is decided bytewise (PREREG Sec.6): SH theta / exp_avg / exp_avg_sq /
global step / Gaussian count must hash-equal between the dense and lazy lanes.
These helpers centralise the canonical serialisation so hashes are stable across
platforms and independent of dict insertion order.
"""

import hashlib
import json

import numpy as np
import torch


def _to_numpy_bytes(t):
    """Contiguous CPU float32 bytes of a tensor (canonical form)."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    return t.detach().to("cpu", copy=False).contiguous().numpy().tobytes()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def hash_tensor(t):
    """SHA256 over the raw little-endian float32 bytes of `t`."""
    return sha256_bytes(_to_numpy_bytes(t))


def hash_tensor_list(tensors):
    """Order-sensitive SHA256 over several tensors (concatenated bytes)."""
    h = hashlib.sha256()
    for t in tensors:
        h.update(_to_numpy_bytes(t))
    return h.hexdigest()


def hash_int(value):
    return sha256_bytes(np.int64(int(value)).tobytes())


def hash_float_list(values):
    """Hash a sequence of Python/numpy floats as fixed 8-byte doubles."""
    arr = np.asarray(list(values), dtype=np.float64)
    return sha256_bytes(arr.tobytes())


def canonical_state_hash(state):
    """Byte hash over a logical-state dict, sorted by key for order-invariance.

    `state` maps str -> tensor. Keys are hashed in sorted order so permutation
    of the dict does not change the digest. Tensor values are hashed as raw
    bytes (no shape prefix) so only the materialised values matter.
    """
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        h.update(key.encode("utf-8"))
        h.update(_to_numpy_bytes(state[key]))
    return h.hexdigest()


def write_json(path, obj):
    """Write JSON deterministically (sorted keys, 2-space indent, trailing NL)."""
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.write("\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
