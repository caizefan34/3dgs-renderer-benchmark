#!/usr/bin/env python3
"""
ReproTopo0 harness — shared utilities (remote: EPIC-05 A100).

ReproTopo0 (FROZEN prereg, packet REPROTOPO0_PACKET_20260930):
execution determinism of a GIVEN 3DGS topology policy (C0 / reference_v1 absgrad).

This module provides:
  - authoritative paths + sha256 helpers (rule #1: full 64-hex identity checks)
  - frozen config load
  - deterministic counter-based RNG (splitmix64 -> uniform -> normal), device-independent
  - logical-id helpers + child-id Hash64
  - canonical full-state byte hash (for determinism stress)
  - JSON emit helper

Nothing here changes any densification/pruning threshold or policy predicate.
"""
import os
import sys
import json
import math
import hashlib
import datetime

import numpy as np
import torch

# ------------------------------------------------------------------ paths ----
REPO = "/root/3dgs-renderer-benchmark"
REF_V1_DIR = os.path.join(REPO, "baseline", "reference_v1")
N1_RUNS = "/root/epic05_n1/runs"
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
SRC_DIR = os.path.join(REPO, "src")
PHASE7_DIR = os.path.join(REPO, "scripts", "epic05", "phase7")
C0_SO = "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
N1_TRAINER = "/root/epic05_n1/publication_trainer_p12_n1snap.py"

CELLS = {
    "room@10000":    {"scene": "room",    "ckpt": os.path.join(N1_RUNS, "n1snap_c0f_s42_room",    "ckpt_10000.pt")},
    "bicycle@10000": {"scene": "bicycle", "ckpt": os.path.join(N1_RUNS, "n1snap_c0f_s42_bicycle", "ckpt_10000.pt")},
}

# authoritative asset hashes (packet AUTHORITATIVE_ASSETS.json / DSH_HANDOFF)
AUTHORITATIVE = {
    "c0_extension_sha256": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
    "core_gsplat_cuda_sha256": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
    "fused_ssim_sha256": "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb",
    "cells_4": {
        "room@10000":    "9aa00f0ddf28b6c097e3dc8f41dbb107cd9bf175750503dc330aa1d6ab579fc4",
        "room@15000":    "b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd",
        "bicycle@10000": "26bf0b959f10f0ce526c496d0dc7ca8e62b0ba93efc83d58b7f9440d58c2f4e9",
        "bicycle@15000": "273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7",
    },
}

# ------------------------------------------------------------- hashing --------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def emit_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path


# ------------------------------------------------------------- config --------
def load_config():
    """Frozen reference_v1 config (imported so thresholds come from the source
    of truth, never hard-coded here)."""
    if REF_V1_DIR not in sys.path:
        sys.path.insert(0, REF_V1_DIR)
    from config import ReferenceV1Config
    return ReferenceV1Config()


# --------------------------------------------------- counter-based RNG -------
# Device-independent: pure integer hashing + float64 inverse-CDF normal.
# Result is bitwise identical on CPU/GPU0..7 -> supports cross-A100 parity.
_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15


def _splitmix64(x):
    x = (x + _GOLDEN) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = z ^ (z >> 31)
    return z & _MASK64


def _key_u64(global_seed, event_index, parent_id, child_index, component):
    k = (int(global_seed) & _MASK64)
    k = (k ^ ((int(event_index) & _MASK64) * 0xD1342543DE94EF95)) & _MASK64
    k = (k ^ ((int(parent_id) & _MASK64) * 0xA256CA6E2C6D6B0D)) & _MASK64
    k = (k ^ ((int(child_index) & _MASK64) * 0x9E3779B97F4A7C15)) & _MASK64
    k = (k ^ ((int(component) & _MASK64) * 0xC2B2AE3D27D4EB4F)) & _MASK64
    return _splitmix64(k)


def counter_unit(global_seed, event_index, parent_id, child_index, component):
    """Deterministic uniform in [0,1) from the counter key."""
    u64 = _key_u64(global_seed, event_index, parent_id, child_index, component)
    u01 = ((u64 >> 11) + 0.5) * (2.0 ** -53)
    return u01


def _ndtri(p):
    """Inverse standard-normal CDF (Moro 1995 rational approx), float64,
    device-independent. Central accuracy ~1e-9."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def counter_normal(global_seed, event_index, parent_id, child_index, component):
    """Deterministic standard-normal sample (python float, float64)."""
    u = counter_unit(global_seed, event_index, parent_id, child_index, component)
    return _ndtri(u)


# --------------------------------------- vectorized counter RNG (numpy uint64) ----
# Device-independent: computed on CPU in numpy uint64 (logical >> shifts, mod-2^64
# wrap) so it is bitwise identical everywhere. Used to sample split-child offsets
# for large clouds without a Python per-row loop. Matches the scalar path exactly.
_MASK64_PY = (1 << 64) - 1


def _splitmix64_np(x):
    M = np.uint64(_MASK64_PY)
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & M
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & M
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & M
    z = z ^ (z >> np.uint64(31))
    return z & M


def _ndtri_np(p):
    """Vectorized inverse standard-normal CDF (Moro 1995), float64.
    Bitwise-mirrors the scalar _ndtri (same coefficients/branch thresholds)."""
    p = np.asarray(p, dtype=np.float64)
    a = np.array([-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                  1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00])
    b = np.array([-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                  6.680131188771972e+01, -1.328068155288572e+01])
    c = np.array([-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                  -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00])
    d = np.array([7.784695709041462e-03, 3.224671290700398e-01,
                  2.445134137142996e+00, 3.754408661907416e+00])
    plow = 0.02425
    phigh = 1 - plow
    out = np.empty_like(p)

    lo = p < plow
    hi = p > phigh
    ce = ~(lo | hi)

    q = np.sqrt(-2.0 * np.log(p[lo]))
    out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    q = np.sqrt(-2.0 * np.log(1.0 - p[hi]))
    out[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    q = p[ce] - 0.5
    r = q * q
    out[ce] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return out


def counter_normal_np(parent_ids, global_seed, event_index, child_index, n_comp=3):
    """Vectorized standard-normal samples for an array of parent ids at a fixed
    child_index, for components 0..n_comp-1. Returns float64 [S, n_comp].
    Equivalent to calling scalar counter_normal per (parent, child_index, comp)."""
    M = np.uint64(_MASK64_PY)
    pid = np.asarray(parent_ids).astype(np.uint64) & M
    k = pid * np.uint64(0) + np.uint64(int(global_seed) & _MASK64_PY)
    k = k ^ np.uint64((int(event_index) & _MASK64_PY) * 0xD1342543DE94EF95 & _MASK64_PY)
    k = k ^ (pid * np.uint64(0xA256CA6E2C6D6B0D))
    k = k ^ np.uint64((int(child_index) & _MASK64_PY) * 0x9E3779B97F4A7C15 & _MASK64_PY)
    cols = []
    for c in range(n_comp):
        kc = k ^ np.uint64((int(c) & _MASK64_PY) * 0xC2B2AE3D27D4EB4F & _MASK64_PY)
        u64 = _splitmix64_np(kc)
        u01 = ((u64 >> np.uint64(11)).astype(np.float64) + 0.5) * (2.0 ** -53)
        cols.append(_ndtri_np(u01))
    return np.stack(cols, axis=1)


# -------------------------------------------------- child-id Hash64 ----------
def child_logical_id(parent_id, event_index, child_index, action_type):
    """Deterministic child id (packet §3.D). Masked to 63 bits so it fits a
    signed int64 torch.long tensor (positive range). action_type in
    {'clone','split'}. Space is 2^63, collision-safe for this workload."""
    at = 0 if action_type == "clone" else 1
    k = (int(parent_id) & _MASK64)
    k = (k ^ ((int(event_index) & _MASK64) * 0x9E3779B97F4A7C15)) & _MASK64
    k = (k ^ ((int(child_index) & _MASK64) * 0xC2B2AE3D27D4EB4F)) & _MASK64
    k = (k ^ (at * 0x165667B19E3779F9)) & _MASK64
    return _splitmix64(k) & ((1 << 63) - 1)


def child_logical_id_np(parent_ids, event_index, child_index, action_type):
    """Vectorized child_logical_id (numpy uint64). Returns int64 array [S]
    (values masked to 63 bits, positive). Matches scalar child_logical_id."""
    M = np.uint64(_MASK64_PY)
    M63 = (np.uint64(1) << np.uint64(63)) - np.uint64(1)
    at = np.uint64(0 if action_type == "clone" else 1)
    pid = np.asarray(parent_ids).astype(np.uint64) & M
    k = pid
    k = k ^ np.uint64((int(event_index) & _MASK64_PY) * 0x9E3779B97F4A7C15 & _MASK64_PY)
    k = k ^ np.uint64((int(child_index) & _MASK64_PY) * 0xC2B2AE3D27D4EB4F & _MASK64_PY)
    k = k ^ (at * np.uint64(0x165667B19E3779F9))
    z = _splitmix64_np(k)
    return (z & M63).astype(np.int64)


# ------------------------------------------- full-state byte hash -----------
def canonical_state_hash(tensors_by_logical_id):
    """Byte hash over logical-state tensors sorted by logical id.

    tensors_by_logical_id: dict {name: tensor[N,...]} sharing row order with
    'logical_id'[N] (int64). Sorting by logical id canonicalizes order so the
    hash is invariant under input-row permutation (packet §5C).
    Returns (sha256_hex, N).
    """
    def _np(t):
        return t.detach().to(torch.float32).cpu().contiguous().numpy()
    lid = tensors_by_logical_id["logical_id"].detach()
    order = torch.argsort(lid)
    h = hashlib.sha256()
    h.update(sha256_bytes(lid[order].to(torch.int64).cpu().numpy().tobytes()).encode())  # lid detached
    for name in sorted(tensors_by_logical_id.keys()):
        if name == "logical_id":
            continue
        t = tensors_by_logical_id[name]
        tb = t[order]
        cb = _np(tb).tobytes()
        h.update(sha256_bytes(name.encode()).encode())
        h.update(str(tuple(tb.shape)).encode())
        h.update(cb)
    return h.hexdigest(), int(lid.shape[0])


def detect_collisions(logical_ids):
    if logical_ids.numel() == 0:
        return 0, 0
    uniq = torch.unique(logical_ids)
    return int(logical_ids.numel() - uniq.numel()), int(uniq.numel())
