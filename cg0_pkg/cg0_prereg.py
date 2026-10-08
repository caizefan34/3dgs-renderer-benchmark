#!/usr/bin/env python3
"""CG0 frozen prereg constants + helpers.

CG0 = Gaussian-as-Compiled-Representation oracle. Optimize/recover geometry
first (oracle depth); compile Gaussians afterward; short exact C0 refinement.

Everything here is frozen by CG0_PREREG.md. No threshold tuning after results.
"""
import os, math, json, hashlib
import torch

# ---- paths (EPIC-05 authoritative) -----------------------------------------
N1_ROOT = "/root/epic05_n1"
SNAP_RUNS = os.path.join(N1_ROOT, "runs")
PROV = os.path.join(N1_ROOT, "artifacts")
REPO = "/root/3dgs-renderer-benchmark"
OUT = "/root/epic05_cg0/out"

# ---- frozen binaries / env -------------------------------------------------
P11_FUSED_SO_SHA256 = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"
C0_EXT_SHA256 = "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11"
CORE_SO_SHA256 = "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"
PY = "/root/miniforge3/envs/gsplat/bin/python"

# ---- cells -----------------------------------------------------------------
CELLS = ("room@15000", "room@25000", "bicycle@15000", "bicycle@25000")
CELL_SCENE = {"room@15000": "room", "room@25000": "room",
              "bicycle@15000": "bicycle", "bicycle@25000": "bicycle"}
CELL_ITER = {"room@15000": 15000, "room@25000": 25000,
             "bicycle@15000": 15000, "bicycle@25000": 25000}

def ckpt_path(scene, it):
    return os.path.join(SNAP_RUNS, f"n1snap_c0f_s42_{scene}", f"ckpt_{it}.pt")

def prov_path(scene):
    return os.path.join(PROV, f"N1_CHECKPOINT_IDENTITY_{scene}.json")

# ---- authoritative 30K C0 training wall (timing.json total_wall_s) ---------
WALL30K = {"room": 507.4639733559452, "bicycle": 857.6165891260607}

# ---- frozen C0 image loss (prereg section 8) -------------------------------
#   0.8 * L1 + 0.2 * (1 - SSIM)
LAMBDA_DSSIM = 0.2

# ---- camera / split semantics ---------------------------------------------
HOLDOUT_MOD = 8          # true-holdout = original index i % 8 == 0
STRIDE = 4               # pixel stride for candidate sampling
N_COMPILE = 64
N_CALIB = 8
N_TRAINVAL = 16
N_HOLDOUT_EVAL = 16
NORMAL_COS_THRESH = 0.90
DEPTH_DISCONT_FRAC = 0.02

# ---- SH basis (standard 3DGS real SH, matches gsplat rasterizer) ----------
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435)
SH_DC_OFFSET = 0.5       # color = 0.5 + sum_k sh_k * B_k(dir)

def sh_basis(dirs):
    """dirs: [...,3] normalized view dirs (from camera to point).
    Returns [...,16] real-SH basis B_k(dir) with C0 folded into B_0."""
    x = dirs[..., 0]; y = dirs[..., 1]; z = dirs[..., 2]
    B = torch.empty(*dirs.shape[:-1], 16, dtype=dirs.dtype, device=dirs.device)
    B[..., 0] = C0
    B[..., 1] = -C1 * y
    B[..., 2] = C1 * z
    B[..., 3] = -C1 * x
    B[..., 4] = C2[0] * x * y
    B[..., 5] = C2[1] * y * z
    B[..., 6] = C2[2] * (2.0 * z * z - x * x - y * y)
    B[..., 7] = C2[3] * x * z
    B[..., 8] = C2[4] * (x * x - y * y)
    B[..., 9] = C3[0] * y * (3.0 * x * x - y * y)
    B[..., 10] = C3[1] * x * y * z
    B[..., 11] = C3[2] * y * (4.0 * z * z - x * x - y * y)
    B[..., 12] = C3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
    B[..., 13] = C3[4] * x * (4.0 * z * z - x * x - y * y)
    B[..., 14] = C3[5] * z * (x * x - y * y)
    B[..., 15] = C3[6] * x * (x * x - 3.0 * y * y)
    return B

def sh_degree_for_count(n):
    if n >= 20: return 3
    if n >= 9: return 2
    if n >= 4: return 1
    return 0

def n_coeffs_for_degree(d):
    return (d + 1) ** 2

# ---- camera split helpers --------------------------------------------------
def stratified(idx, k):
    """midpoint-stratified: p = floor((i+0.5)*N/k), deterministic (vjs0)."""
    N = len(idx); used = set(); out = []
    for i in range(k):
        p = int(math.floor((i + 0.5) * N / k))
        while p in used or p >= N:
            p += 1
        used.add(p); out.append(p)
    return out

def camera_semantics(n):
    holdout = [i for i in range(n) if i % HOLDOUT_MOD == 0]
    train = [i for i in range(n) if i % HOLDOUT_MOD != 0]
    assert set(train).isdisjoint(set(holdout))
    return train, holdout

def build_split(n):
    """Deterministic disjoint split per prereg section 2."""
    train, holdout = camera_semantics(n)
    comp = stratified(train, N_COMPILE)
    comp_set = set(comp)
    rem1 = [i for i in train if i not in comp_set]
    calib = stratified(rem1, N_CALIB)
    calib_set = set(calib)
    rem2 = [i for i in rem1 if i not in calib_set]
    trainval = stratified(rem2, N_TRAINVAL)
    nh = min(N_HOLDOUT_EVAL, len(holdout))
    holdout_eval = stratified(holdout, nh)
    return {
        "n_total": n, "n_train": len(train), "n_holdout": len(holdout),
        "compile64": sorted(comp), "calib8": sorted(calib),
        "trainval16": sorted(trainval), "holdout16": sorted(holdout_eval),
    }

# ---- math helpers ----------------------------------------------------------
def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))

def mat_to_quat(R):
    """3x3 rotation matrix (torch tensor) -> quaternion [w,x,y,z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = torch.stack([w, x, y, z])
    return q / (q.norm() + 1e-12)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jdefault)
    print(f"[cg0] wrote {path}", flush=True)

def _jdefault(o):
    if isinstance(o, (torch.Tensor,)): return o.detach().cpu().tolist()
    if hasattr(o, "item"): return o.item()
    return str(o)
