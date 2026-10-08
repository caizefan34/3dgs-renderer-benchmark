#!/usr/bin/env python3
"""PSC-P0 probe: execute the frozen preregistration on EPIC-05 (A100, GPU1).

Writes all required artifacts to RUN_DIR and prints a final JSON summary.
The frozen analyzer psc_p0_analyze.py owns the mechanical verdict.
"""
import sys, os, math, json, time, hashlib, platform, datetime
import fcntl
import numpy as np
import torch
import torch.nn.functional as F

REPO = "/root/3dgs-renderer-benchmark"
RUN_DIR = "/root/3dgs-renderer-benchmark/PSC_P0_RUN"
REFERENCE_V1_DIR = os.path.join(REPO, "baseline", "reference_v1")
PHASE7_DIR = os.path.join(REPO, "scripts", "epic05", "phase7")
SRC_DIR = os.path.join(REPO, "src")
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
C0_SO = "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CKPT = "/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_25000.pt"
TRAINER = "/root/epic05_p12/publication_trainer_p12.py"
ANALYZER = os.path.join(RUN_DIR, "psc_p0_analyze.py")

EXPECTED = {
    "checkpoint": ["b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d", CKPT],
    "c0_extension": ["d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11", C0_SO],
    "core_gsplat_cuda": ["935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e", CORE_SO],
    "trainer": ["675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5", TRAINER],
}

# --- DOCUMENTED DEVIATION FROM FROZEN PREREG (c0_extension identity) ---
# The prereg constant for c0_extension in PSC_P0_PREREG_v1.md is 63 hex chars,
# which is NOT a valid SHA-256 (always 64). No file can ever match it, so the
# gate as written can never pass for C0. The on-disk C0 extension is
# independently verified as the genuine frozen artifact by the authoritative
# P12 source-identity record (P12_SOURCE_IDENTITY.json -> c0_lineage.ext_so.
# sha256), which records the exact 64-char hash present on disk. We therefore
# bind the c0_extension identity check to that independent authoritative value
# at runtime and record this as a documented deviation. This corrects a malformed
# prereg constant; it is NOT a FAIL->PASS upgrade of a genuine mismatch.
P12_IDENTITY = "/root/epic05_p12/artifacts/P12_SOURCE_IDENTITY.json"
IDENTITY_DEVIATIONS = {
    "c0_extension": {
        "prereg_constant": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11",
        "prereg_constant_len": 63,
        "authoritative_source": "P12_SOURCE_IDENTITY.json:c0_lineage.ext_so.sha256",
        "reason": "prereg constant is 63 hex chars (invalid SHA-256 length; impossible to match any file); on-disk artifact independently verified genuine by the P12 source-identity record",
    },
}

def authoritative_c0_hash():
    import json as _json
    with open(P12_IDENTITY) as f:
        d = _json.load(f)
    return d["c0_lineage"]["ext_so"]["sha256"]

def log(msg):
    print(msg, flush=True)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, torch.Tensor): return o.detach().cpu().tolist()
    return str(o)

def write_json(name, obj):
    path = os.path.join(RUN_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    log("WROTE " + path)
    return path

def write_text(name, text):
    path = os.path.join(RUN_DIR, name)
    with open(path, "w") as f:
        f.write(text)
    log("WROTE " + path)
    return path

def load_ext_module(name, so_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

def bootstrap_c0():
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    os.environ["HIGS_BWD_H8_MR"] = "1"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    os.environ.pop("HIGS_DISABLE_F9", None)
    sys.path.insert(0, C0_WT)
    sys.path.insert(0, SCENE_CUDA_DIR)
    core = load_ext_module("gsplat_cuda", CORE_SO)
    sys.modules["gsplat.csrc"] = core
    exp = load_ext_module("experimental_gaussian_render_inference_scene_cuda", C0_SO)
    sys.path.insert(0, PHASE7_DIR)
    sys.path.insert(0, SRC_DIR)
    sys.path.insert(0, REFERENCE_V1_DIR)
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE
    return core, exp

class SepSSIM:
    def __init__(self, device="cuda", window_size=11, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32, device=device)
        coords -= window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2
    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + 0.01 ** 2) * (2 * spt + 0.03 ** 2) / \
                   ((mu_p2 + mu_t2 + 0.01 ** 2) * (sp2 + st2 + 0.03 ** 2))
        return 1.0 - ssim_map.mean()

# ---------- geometry helpers ----------
def conics_to_Sigma(conics):
    G = conics.shape[0]
    # Numerical guard: a few cohort Gaussians can yield a singular 2D precision
    # matrix (degenerate/overflowed 2D covariance in a given view). Such a
    # Gaussian has no invertible 2D covariance, so Sigma is undefined; we floor
    # the conic diagonal to keep Sigma finite. This only affects the measure-zero
    # set of exactly-singular Gaussians and does not alter the frozen science
    # (cameras, cohorts, thresholds, or the R_int/LOO estimators for well-posed
    # Gaussians). The count is logged for the contamination report. Clamping is
    # done out-of-place on the conic values BEFORE building C to avoid any
    # in-place write to a tensor that autograd needs for the backward pass.
    a = conics[:, 0].clamp_min(1e-12)
    b = conics[:, 1]
    c = conics[:, 2].clamp_min(1e-12)
    n_singular = int(((conics[:, 0] <= 0) | (conics[:, 2] <= 0)).sum().item())
    if n_singular > 0:
        log("conics_to_Sigma: floored " + str(n_singular) + " singular conic(s) in this view")
    C = torch.zeros(G, 2, 2, device=conics.device, dtype=conics.dtype)
    C[:, 0, 0] = a; C[:, 0, 1] = b
    C[:, 1, 0] = b; C[:, 1, 1] = c
    return torch.linalg.inv(C)

def mat_sqrt_inv(Sigma):
    ev, V = torch.linalg.eigh(Sigma)
    ev = ev.clamp_min(1e-12)
    return (V * ev.rsqrt().unsqueeze(1)) @ V.transpose(1, 2)

def build_W(P):
    G = P.shape[0]
    p = P[:, 0, 0]; q = P[:, 0, 1]; r = P[:, 1, 1]
    W = torch.zeros(G, 5, 5, device=P.device, dtype=P.dtype)
    W[:, 0, 0] = p; W[:, 0, 1] = q; W[:, 1, 0] = q; W[:, 1, 1] = r
    W[:, 2, 2] = p * p; W[:, 2, 3] = 2 * p * q; W[:, 2, 4] = q * q
    W[:, 3, 2] = p * q; W[:, 3, 3] = q * q + p * r; W[:, 3, 4] = q * r
    W[:, 4, 2] = q * q; W[:, 4, 3] = 2 * q * r; W[:, 4, 4] = r * r
    return W

def compute_J_cohort(theta, viewmats, Ks, W, H):
    from gsplat.cuda._wrapper import fully_fused_projection
    G = theta.shape[0]
    radii, m2d, depths, conics, comp = fully_fused_projection(
        theta[:, 0:3], None, theta[:, 3:7], theta[:, 7:10], viewmats, Ks, W, H,
        eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
        packed=False, calc_compensations=False)
    m2d0 = m2d[0]; conics0 = conics[0]
    # Validity mask: a (gaussian, view) pair is well-posed only when the gaussian
    # is actually rendered in that view (radius > 0) AND its 2D precision is PSD
    # (det > 0). Culled Gaussians (behind camera / off-screen / zero radius) have an
    # UNDEFINED projected state z_v^G = (mu_2d, Sigma_2d). Diagnostic confirmed
    # vis & ~psd == 0 in all 8 views, so radius > 0 is the operative cull test.
    # For invalid pairs we substitute the identity conic (out-of-place, graph-safe
    # via torch.where so valid entries stay connected to theta for the Jacobian)
    # and zero their Jacobian rows below; collect_cohort zeros their d rows too.
    # Invalid pairs then contribute exactly 0 to every energy-weighted sum, the
    # correct "no projected state" limit. Documented, reversible deviation affecting
    # only undefined pairs; frozen cohort/cameras/thresholds and the R_int/LOO
    # estimators for well-posed pairs are unchanged.
    a_c = conics0[:, 0]; b_c = conics0[:, 1]; c_c = conics0[:, 2]
    det_c = a_c * c_c - b_c * b_c
    valid = (radii[0] > 0).all(dim=-1) & (det_c > 0)
    n_invalid = int((~valid).sum().item())
    if n_invalid > 0:
        log("compute_J_cohort: " + str(n_invalid) + "/" + str(G) + " culled/undefined (gaussian,view) pairs excluded (J,d zeroed)")
        ident = torch.stack([torch.ones_like(a_c), torch.zeros_like(b_c), torch.ones_like(c_c)], dim=1)
        conics0 = torch.where(valid.unsqueeze(-1), conics0, ident)
    Sigma = conics_to_Sigma(conics0)
    # z = (mu_2d[x,y], Sigma_00, Sigma_01, Sigma_11): 2 mean + 3 unique symmetric Sigma entries.
    z = torch.cat([m2d0, Sigma[:, 0, 0:1], Sigma[:, 0, 1:2], Sigma[:, 1, 1:2]], dim=1)
    J = torch.zeros(G, 5, 10, device=theta.device, dtype=theta.dtype)
    for k in range(5):
        (gk,) = torch.autograd.grad(z[:, k].sum(), theta, retain_graph=True)
        J[:, k, :] = gk
    P = mat_sqrt_inv(Sigma)
    Wt = build_W(P)
    J_tilde = torch.einsum("gij,gjk->gik", Wt, J)
    if n_invalid > 0:
        J_tilde = J_tilde.clone()
        J_tilde[~valid] = 0.0
    return J_tilde, Sigma, P, m2d0, conics0, valid

def grad_conic_to_Sigma(conics0, g_conic):
    # conics0: [G,3] conic values (a,b,c) with C=[[a,b],[b,c]]; Sigma = inv(C).
    # g_conic: [G,3] = dL/d(conic).  Want g_Sigma = dL/d(vech(Sigma)).
    # s = vech(inv(C)) = f(conic);  dL/ds = (Jf^T)^{-1} g_conic,  Jf = ds/dconic.
    # Analytic conic->vech(Sigma) Jacobian (matches conics_to_Sigma's clamped
    # conic diagonal). With C=[[a,b],[b,c]], D=det(C)=ac-b^2,
    # Sigma=inv(C)=[[c/D,-b/D],[-b/D,a/D]], s=[Sigma00,Sigma01,Sigma11]:
    #   ds/da = [-c^2, 2bc, -b^2]/D^2
    #   ds/db = [bc, -(ac+b^2), ab]/D^2
    #   ds/dc = [-b^2, 2ab, -a^2]/D^2
    # Using the analytic form (instead of autograd + linalg.solve) keeps this
    # well-posed for the same measure-zero degenerate Gaussians that conics_to_Sigma
    # floors: D is bounded away from 0 by the conic clamp, so Jf is invertible and
    # the (Jf^T)^{-1} g_conic solve cannot raise a singular-matrix error. This does
    # not change the frozen science; it only makes the gradient transport robust.
    a = conics0[:, 0].clamp_min(1e-12)
    b = conics0[:, 1]
    c = conics0[:, 2].clamp_min(1e-12)
    D = (a * c - b * b).clamp_min(1e-24)
    invD2 = 1.0 / (D * D)
    Jf = torch.zeros(conics0.shape[0], 3, 3, device=conics0.device, dtype=conics0.dtype)
    Jf[:, 0, 0] = -c * c * invD2; Jf[:, 0, 1] = 2.0 * b * c * invD2; Jf[:, 0, 2] = -b * b * invD2
    Jf[:, 1, 0] = b * c * invD2;  Jf[:, 1, 1] = -(a * c + b * b) * invD2; Jf[:, 1, 2] = a * b * invD2
    Jf[:, 2, 0] = -b * b * invD2; Jf[:, 2, 1] = 2.0 * a * b * invD2; Jf[:, 2, 2] = -a * a * invD2
    g_Sigma = torch.linalg.solve(Jf.transpose(1, 2), g_conic.unsqueeze(-1)).squeeze(-1)
    return g_Sigma

def render_full(means_l, quats_l, scales_l, colors, opac_b, viewmats, Ks, W, H, ssim_fn, gt, absgrad):
    from gsplat.cuda._wrapper import (
        fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels)
    radii, m2d, depths, conics, comp = fully_fused_projection(
        means_l, None, quats_l, scales_l, viewmats, Ks, W, H,
        eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
        packed=False, calc_compensations=False)
    tile = 16; tw = math.ceil(W / tile); th = math.ceil(H / tile)
    tpg, isect_ids, flatten_ids = isect_tiles(
        m2d, radii, depths, tile, tw, th, packed=False, n_images=1,
        conics=conics, opacities=opac_b)
    isect_offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(1, th, tw)
    render, alphas = rasterize_to_pixels(
        m2d, conics, colors, opac_b, W, H, tile, isect_offsets, flatten_ids,
        backgrounds=None, packed=False, absgrad=absgrad)
    img = render[0].clamp(0, 1)
    L1 = F.l1_loss(img, gt)
    dssim = ssim_fn(img, gt)
    loss = 0.8 * L1 + 0.2 * dssim
    return loss, m2d, conics

def main():
    t_start = time.time()
    os.makedirs(RUN_DIR, exist_ok=True)
    # Single-instance lock: this probe uses a frozen CUDA extension and the shared
    # GPU; two concurrent runs corrupt each other's CUDA state (observed as NaN in
    # PSC_METRICS.json). Refuse to start if another probe is already running.
    lock_path = os.path.join(RUN_DIR, ".psc_p0_probe.lock")
    _lock_fp = open(lock_path, "a+")
    try:
        _rc = fcntl.flock(_lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except AttributeError:
        _lock_fp = None  # fcntl unavailable (non-POSIX); skip locking
    except OSError:
        if _lock_fp is not None:
            _lock_fp.close()
        log("ABORT: another psc_p0_probe run holds the lock; refusing to start (avoids GPU state corruption).")
        return
    _lock_fp.seek(0); _lock_fp.truncate(); _lock_fp.write(str(os.getpid())); _lock_fp.flush()
    log("=== PSC-P0 probe start " + datetime.datetime.now().isoformat() + " ===")

    identities = {}
    deviations = []
    all_match = True
    for key, (exp, path) in EXPECTED.items():
        actual = sha256_file(path) if os.path.exists(path) else "MISSING"
        effective_exp = exp
        if key in IDENTITY_DEVIATIONS:
            dev = IDENTITY_DEVIATIONS[key]
            auth = authoritative_c0_hash()
            effective_exp = auth
            deviations.append({
                "identity": key,
                "prereg_constant": dev["prereg_constant"],
                "prereg_constant_len": dev["prereg_constant_len"],
                "authoritative_source": dev["authoritative_source"],
                "authoritative_value_used": auth,
                "on_disk_actual": actual,
                "on_disk_matches_authoritative": (actual == auth),
                "reason": dev["reason"],
            })
        match = (actual == effective_exp)
        all_match = all_match and match
        identities[key] = {"expected": effective_exp, "actual": actual, "path": path, "match": match}
        if key in IDENTITY_DEVIATIONS:
            identities[key]["prereg_constant"] = IDENTITY_DEVIATIONS[key]["prereg_constant"]
            identities[key]["deviation"] = True
    identities["all_match"] = all_match
    identities["deviations"] = deviations
    write_json("FROZEN_IDENTITIES.json", identities)
    if not all_match:
        log("IDENTITY_FAIL: one or more hashes mismatched. Stopping.")
        verdict = {"schema_version": "PSC-P0-v1", "verdict": "PSC_P0_IDENTITY_FAIL",
                   "reason": "frozen identity hash mismatch", "identities": identities}
        write_json("PSC_P0_VERDICT.json", verdict)
        write_text("PSC_P0_REPORT.md", "# PSC-P0 Report\n\nSTATUS: IDENTITY_FAIL\n\nFrozen identity hashes did not match. Execution halted before any metric computation.\n")
        finalize_manifest()
        log(json.dumps({"verdict": "PSC_P0_IDENTITY_FAIL"}))
        return

    if not os.path.exists(ANALYZER):
        log("MISSING analyzer: " + ANALYZER)
    env = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "ngpu": torch.cuda.device_count(),
        "device_used": "CUDA (CUDA_VISIBLE_DEVICES)",
        "worktree": C0_WT,
        "checkpoint": CKPT,
        "trainer": TRAINER,
    }
    write_json("ENVIRONMENT.json", env)

    bootstrap_c0()
    from gaussian_model import GaussianModel
    from config import ReferenceV1Config
    from dataset import GTDataset
    from gsplat.cuda._wrapper import (
        fully_fused_projection, isect_tiles, isect_offset_encode,
        rasterize_to_pixels, spherical_harmonics)
    log("IMPORTS_OK")

    config = ReferenceV1Config()
    config.scene = "room"
    config.repo_root = REPO
    dataset = GTDataset(scene="room", repo_root=REPO, resolution="1080p", device="cuda", background="black")
    log("N_CAMS " + str(len(dataset)))

    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = GaussianModel(max_sh_degree=3)
    # training_setup() (called inside restore) requires a subscriptable dict of
    # optimization params, matching the canonical P12 trainer pattern
    # (publication_trainer_p12.py L428-438). Build it from the frozen
    # ReferenceV1Config attributes; no threshold/value is altered.
    opt_dict = {
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    }
    model.restore(state, opt_dict)
    means = model.get_xyz
    quats = model.get_rotation
    scales = model.get_scaling
    opac = model.get_opacity
    shs = model.get_features
    N = means.shape[0]
    log("N " + str(N) + " active_sh " + str(model.active_sh_degree))
    ssim_fn = SepSSIM(device="cuda")

    centers = []
    for i in range(len(dataset)):
        cam = dataset.get_camera(i)
        centers.append(cam.camera_center.detach().cpu())
    centers = torch.stack(centers)
    seed = 19
    selected = [seed]
    sel_centers = [centers[seed]]
    while len(selected) < 8:
        sc = torch.stack(sel_centers)
        d = torch.cdist(centers, sc)
        dmin = d.min(dim=1).values
        for s in selected:
            dmin[s] = -1.0
        nxt = int(dmin.argmax().item())
        selected.append(nxt)
        sel_centers.append(centers[nxt])
    cameras = selected
    cam_records = []
    for cid in cameras:
        cam = dataset.get_camera(cid)
        cam_records.append({
            "valid_index": int(cid),
            "image_name": getattr(cam, "image_name", None),
            "camera_center": cam.camera_center.detach().cpu().tolist(),
        })
    camera_set = {
        "seed": seed,
        "method": "camera-center farthest-point sampling",
        "n_cameras": len(cameras),
        "cameras": cameras,
        "records": cam_records,
        "nested_B": {"2": cameras[:2], "4": cameras[:4], "8": cameras[:8]},
    }
    write_json("CAMERA_SET.json", camera_set)
    log("CAMERAS " + str(cameras))

    means_l = means.detach().clone().requires_grad_(True)
    quats_l = quats.detach().clone().requires_grad_(True)
    scales_l = scales.detach().clone().requires_grad_(True)

    cam, gt = dataset.get_item(19)
    W = int(cam.image_width); H = int(cam.image_height)
    viewmats = cam.viewmatrix.unsqueeze(0)
    Ks = cam.K.unsqueeze(0)
    with torch.no_grad():
        radii0, _, _, _, _ = fully_fused_projection(
            means, None, quats, scales, viewmats, Ks, W, H,
            eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
            packed=False, calc_compensations=False)
    masks0 = (radii0 > 0).all(dim=-1)[0]
    colors0 = torch.clamp(spherical_harmonics(3, means - cam.camera_center, shs, masks=masks0).unsqueeze(0) + 0.5, 0.0)
    opac_b = opac.unsqueeze(0).expand(1, -1).contiguous()

    t_means = means_l.detach().clone().requires_grad_(True)
    t_quats = quats_l.detach().clone().requires_grad_(True)
    t_scales = scales_l.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([t_means, t_quats, t_scales], lr=1e-3)

    def full_step():
        optimizer.zero_grad()
        radii, m2d, depths, conics, comp = fully_fused_projection(
            t_means, None, t_quats, t_scales, viewmats, Ks, W, H,
            eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
            packed=False, calc_compensations=False)
        tile = 16; tw = math.ceil(W / tile); th = math.ceil(H / tile)
        tpg, isect_ids, flatten_ids = isect_tiles(
            m2d, radii, depths, tile, tw, th, packed=False, n_images=1,
            conics=conics, opacities=opac_b)
        isect_offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(1, th, tw)
        render, alphas = rasterize_to_pixels(
            m2d, conics, colors0, opac_b, W, H, tile, isect_offsets, flatten_ids,
            backgrounds=None, packed=False, absgrad=False)
        img = render[0].clamp(0, 1)
        loss = 0.8 * F.l1_loss(img, gt) + 0.2 * ssim_fn(img, gt)
        gmu, gcon = torch.autograd.grad(loss, [m2d, conics], retain_graph=True)
        zscalar = (m2d * gmu).sum() + (conics * gcon).sum()
        gtheta = torch.autograd.grad(zscalar, [t_means, t_quats, t_scales], retain_graph=True)
        t_means.grad = gtheta[0]; t_quats.grad = gtheta[1]; t_scales.grad = gtheta[2]
        optimizer.step()

    WARMUP = 10; REPS = 100
    for _ in range(WARMUP):
        full_step()
    torch.cuda.synchronize()
    init_params = [t.detach().clone() for t in (t_means, t_quats, t_scales)]
    t_proj = []; t_screen = []; t_vjp = []; t_opt = []
    for _ in range(REPS):
        for t, v in zip((t_means, t_quats, t_scales), init_params):
            t.data.copy_(v)
        optimizer.zero_grad()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e2 = torch.cuda.Event(True); e3 = torch.cuda.Event(True)
        e4 = torch.cuda.Event(True); e5 = torch.cuda.Event(True)
        e6 = torch.cuda.Event(True); e7 = torch.cuda.Event(True)
        torch.cuda.synchronize()
        e0.record()
        radii, m2d, depths, conics, comp = fully_fused_projection(
            t_means, None, t_quats, t_scales, viewmats, Ks, W, H,
            eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
            packed=False, calc_compensations=False)
        e1.record()
        e2.record()
        tile = 16; tw = math.ceil(W / tile); th = math.ceil(H / tile)
        tpg, isect_ids, flatten_ids = isect_tiles(
            m2d, radii, depths, tile, tw, th, packed=False, n_images=1,
            conics=conics, opacities=opac_b)
        isect_offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(1, th, tw)
        render, alphas = rasterize_to_pixels(
            m2d, conics, colors0, opac_b, W, H, tile, isect_offsets, flatten_ids,
            backgrounds=None, packed=False, absgrad=False)
        img = render[0].clamp(0, 1)
        loss = 0.8 * F.l1_loss(img, gt) + 0.2 * ssim_fn(img, gt)
        gmu, gcon = torch.autograd.grad(loss, [m2d, conics], retain_graph=True)
        e3.record()
        e4.record()
        zscalar = (m2d * gmu).sum() + (conics * gcon).sum()
        gtheta = torch.autograd.grad(zscalar, [t_means, t_quats, t_scales], retain_graph=True)
        t_means.grad = gtheta[0]; t_quats.grad = gtheta[1]; t_scales.grad = gtheta[2]
        e5.record()
        e6.record()
        optimizer.step()
        e7.record()
        torch.cuda.synchronize()
        t_proj.append(e0.elapsed_time(e1))
        t_screen.append(e2.elapsed_time(e3))
        t_vjp.append(e4.elapsed_time(e5))
        t_opt.append(e6.elapsed_time(e7))
    t_proj_a = np.array(t_proj); t_screen_a = np.array(t_screen)
    t_vjp_a = np.array(t_vjp); t_opt_a = np.array(t_opt)
    T_screen = float(t_screen_a.mean())
    T_proj = float(t_proj_a.mean())
    T_vjp = float(t_vjp_a.mean())
    T_opt = float(t_opt_a.mean())
    T_global = T_proj + T_vjp + T_opt
    T_native = T_screen + T_global
    f_global = T_global / T_native if T_native > 0 else float("nan")
    S_max = 1.0 / (1.0 - f_global) if f_global < 1.0 else float("inf")
    S_upper = {}
    for K in (2, 4, 8):
        S_upper[str(K)] = T_native / (T_screen + T_global / K)
    timing_summary = {
        "T_screen_ms": T_screen, "T_proj_ms": T_proj, "T_vjp_ms": T_vjp,
        "T_opt_ms": T_opt, "T_global_ms": T_global, "T_native_ms": T_native,
        "f_global": f_global, "S_max": S_max, "S_upper": S_upper,
        "reps": REPS, "warmup": WARMUP,
        "std": {"T_screen_ms": float(t_screen_a.std()), "T_global_ms": float((t_proj_a + t_vjp_a + t_opt_a).std())},
        "camera": 19, "note": "T_global=T_proj+T_vjp+T_opt; colors precomputed (SH excluded from T_screen)",
    }
    write_json("TIMING_RAW.json", {"T_proj_ms": t_proj, "T_screen_ms": t_screen, "T_vjp_ms": t_vjp, "T_opt_ms": t_opt})
    write_json("TIMING_SUMMARY.json", timing_summary)
    log("TIMING T_screen=%.3f T_global=%.3f T_native=%.3f f_global=%.3f S_max=%.3f S_upper8=%.3f" %
        (T_screen, T_global, T_native, f_global, S_max, S_upper["8"]))

    view_data = []
    contrib = torch.zeros(N, device="cuda")
    for vid, cid in enumerate(cameras):
        # Fresh leaf tensors + opacity batch per view: each view's projection
        # graph must be independent so loss.backward() does not traverse a
        # graph already freed by the previous view. Per-view m2d/conics grads
        # are captured immediately below, so this does not alter the science.
        means_l = means.detach().clone().requires_grad_(True)
        quats_l = quats.detach().clone().requires_grad_(True)
        scales_l = scales.detach().clone().requires_grad_(True)
        opac_b = opac.unsqueeze(0).expand(1, -1).contiguous()
        cam, gt = dataset.get_item(cid)
        Wv = int(cam.image_width); Hv = int(cam.image_height)
        vm = cam.viewmatrix.unsqueeze(0); K = cam.K.unsqueeze(0)
        with torch.no_grad():
            r0, _, _, _, _ = fully_fused_projection(means, None, quats, scales, vm, K, Wv, Hv,
                eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False, calc_compensations=False)
            m0 = (r0 > 0).all(dim=-1)[0]
        # colors/opac are detached here: this loop only needs m2d.grad / m2d.absgrad
        # (the rendering-contribution proxy), which flow through m2d <- fresh per-view
        # leaves (means_l/quats_l/scales_l). Detaching colors/opac keeps each view's
        # autograd graph independent of the shared model params (means/shs/opac) so
        # view N's loss.backward() does not free a graph reused by view N+1. The forward
        # values are identical, so the loss and dL/d(m2d) are unchanged (scientifically neutral).
        colors_v = torch.clamp(spherical_harmonics(3, means - cam.camera_center, shs, masks=m0).detach().unsqueeze(0) + 0.5, 0.0)
        opac_b = opac.detach().unsqueeze(0).expand(1, -1).contiguous()
        loss, m2d, conics = render_full(means_l, quats_l, scales_l, colors_v, opac_b, vm, K, Wv, Hv, ssim_fn, gt, absgrad=True)
        m2d.retain_grad()
        conics.retain_grad()
        loss.backward()
        if m2d.absgrad is not None:
            ag = m2d.absgrad
        else:
            ag = m2d.grad.abs()
        contrib += ag[0].sum(dim=-1)
        gmu_full = m2d.grad[0].detach().clone()
        gcon_full = conics.grad[0].detach().clone()
        view_data.append({"cid": cid, "W": Wv, "H": Hv, "vm": vm, "K": K, "gt": gt,
                          "gmu": gmu_full, "gcon": gcon_full, "cam_center": cam.camera_center.detach().clone()})
        log("VIEW " + str(vid) + " cid=" + str(cid) + " loss=" + str(float(loss)) + " collected")

    topk = torch.topk(contrib, 4096).indices
    cohort_primary = sorted(topk.tolist())
    vis_idx = torch.nonzero(contrib > 0, as_tuple=False).squeeze(-1)
    if vis_idx.numel() >= 4096:
        perm = torch.randperm(vis_idx.numel(), device="cuda")[:4096]
        cohort_diag = sorted(vis_idx[perm].tolist())
    else:
        cohort_diag = sorted(vis_idx.tolist())
    write_json("COHORT.json", {"primary": cohort_primary, "diagnostic": cohort_diag,
                               "contribution_proxy": "accumulated |means2d.absgrad| over 8 frozen views"})
    log("COHORT primary=" + str(len(cohort_primary)) + " diag=" + str(len(cohort_diag)))

    def collect_cohort(cohort):
        cid_t = torch.tensor(cohort, device="cuda", dtype=torch.long)
        J_per_view = []; d_per_view = []
        for vid in range(len(cameras)):
            vd = view_data[vid]
            vm = vd["vm"]; K = vd["K"]
            gmu_c = vd["gmu"][cid_t]
            gcon_c = vd["gcon"][cid_t]
            theta = torch.cat([means[cid_t], quats[cid_t], scales[cid_t]], dim=1).detach().clone().requires_grad_(True)
            J_tilde, Sigma, P, m2d0, conics0, valid = compute_J_cohort(theta, vm, K, vd["W"], vd["H"])
            # Zero the projected-state gradient d for culled/undefined (gaussian,view)
            # pairs so they contribute exactly 0 to every energy-weighted sum (the
            # correct "no projected state" limit; matches the zeroed J rows). This
            # does not alter the frozen science for well-posed pairs.
            gmu_c = gmu_c * valid.unsqueeze(-1).to(gmu_c.dtype)
            gcon_c = gcon_c * valid.unsqueeze(-1).to(gcon_c.dtype)
            g_Sigma = grad_conic_to_Sigma(conics0.detach(), gcon_c)
            d_raw = -torch.cat([gmu_c, g_Sigma], dim=1)
            Wt = build_W(P)
            d_white = torch.einsum("gij,gj->gi", Wt, d_raw)
            J_per_view.append(J_tilde.detach())
            d_per_view.append(d_white.detach())
            log("  cohort view " + str(vid) + " done")
        return torch.stack(J_per_view, dim=0), torch.stack(d_per_view, dim=0)

    log("Collecting PRIMARY cohort...")
    J_p, d_p = collect_cohort(cohort_primary)
    log("Collecting DIAGNOSTIC cohort...")
    J_d, d_d = collect_cohort(cohort_diag)

    def r_int_and_loo(J_view, d_view):
        G = J_view.shape[1]
        r_energy = {}
        r_per_g = {}
        for B in (2, 4, 8):
            Jb = J_view[:B].permute(1, 0, 2, 3).reshape(G, B * 5, 10)
            db = d_view[:B].permute(1, 0, 2).reshape(G, B * 5)
            Jp = torch.linalg.pinv(Jb)
            dhat = torch.einsum("gij,gj->gi", Jb, torch.einsum("gij,gj->gi", Jp, db))
            num = (dhat ** 2).sum(dim=1)
            den = (db ** 2).sum(dim=1).clamp_min(1e-30)
            rpg = num / den
            r_energy[str(B)] = float(num.sum() / den.sum())
            r_per_g[str(B)] = rpg.detach().cpu().numpy()
        C = torch.zeros(8, G, device="cuda")
        Wt = torch.zeros(8, G, device="cuda")
        for h in range(8):
            others = [v for v in range(8) if v != h]
            Jo = J_view[others].permute(1, 0, 2, 3).reshape(G, 7 * 5, 10)
            do = d_view[others].permute(1, 0, 2).reshape(G, 7 * 5)
            Jop = torch.linalg.pinv(Jo)
            dtheta = torch.einsum("gij,gj->gi", Jop, do)
            Jh = J_view[h]
            pred = torch.einsum("gij,gj->gi", Jh, dtheta)
            dh = d_view[h]
            cos = (pred * dh).sum(dim=1) / (pred.norm(dim=1) * dh.norm(dim=1) + 1e-30)
            w = (dh ** 2).sum(dim=1)
            C[h] = cos; Wt[h] = w
        P_hold_pos = float((Wt * (C > 0).float()).sum() / Wt.sum())
        P_hold_pos_unw = float((C > 0).float().mean())
        median_C = float(C.median().item())
        return r_energy, r_per_g, {"P_hold_pos": P_hold_pos, "P_hold_pos_unweighted": P_hold_pos_unw, "median_C_hold": median_C}, C, Wt

    log("Computing R_int / LOO PRIMARY...")
    re_p, rpg_p, loo_p, C_p, W_p = r_int_and_loo(J_p, d_p)
    log("Computing R_int / LOO DIAGNOSTIC...")
    re_d, rpg_d, loo_d, C_d, W_d = r_int_and_loo(J_d, d_d)

    np.savez(os.path.join(RUN_DIR, "PSC_RAW_PRIMARY.npz"),
             cohort_ids=np.array(cohort_primary), camera_ids=np.array(cameras),
             J_per_view=J_p.cpu().numpy(), d_per_view=d_p.cpu().numpy(),
             C_hold=C_p.cpu().numpy(), weights=W_p.cpu().numpy(),
             **{("R_int_" + k): v for k, v in rpg_p.items()})
    np.savez(os.path.join(RUN_DIR, "PSC_RAW_DIAGNOSTIC.npz"),
             cohort_ids=np.array(cohort_diag), camera_ids=np.array(cameras),
             J_per_view=J_d.cpu().numpy(), d_per_view=d_d.cpu().numpy(),
             C_hold=C_d.cpu().numpy(), weights=W_d.cpu().numpy(),
             **{("R_int_" + k): v for k, v in rpg_d.items()})
    log("NPZ written")

    metrics = {
        "timing": {
            "S_upper": S_upper,
            "f_global": f_global, "S_max": S_max,
            "T_screen_ms": T_screen, "T_global_ms": T_global, "T_native_ms": T_native,
        },
        "primary": {
            "R_int_energy": re_p,
            "leave_one_out": {"8": {"P_hold_pos": loo_p["P_hold_pos"],
                                     "P_hold_pos_unweighted": loo_p["P_hold_pos_unweighted"],
                                     "median_C_hold": loo_p["median_C_hold"]}},
        },
        "diagnostic": {
            "R_int_energy": re_d,
            "leave_one_out": {"8": {"P_hold_pos": loo_d["P_hold_pos"],
                                     "median_C_hold": loo_d["median_C_hold"]}},
        },
        "cohort": {"primary_size": len(cohort_primary), "diagnostic_size": len(cohort_diag)},
    }
    write_json("PSC_METRICS.json", metrics)

    import subprocess
    analyzer_log = ""
    verdict_obj = None
    if os.path.exists(ANALYZER):
        try:
            r = subprocess.run([sys.executable, ANALYZER, os.path.join(RUN_DIR, "PSC_METRICS.json"),
                                "--out", os.path.join(RUN_DIR, "PSC_P0_VERDICT.json")],
                               capture_output=True, text=True)
            analyzer_log = r.stdout + "\n--STDERR--\n" + r.stderr
            log("ANALYZER_RC=" + str(r.returncode))
            with open(os.path.join(RUN_DIR, "ANALYZER_STDOUT.txt"), "w") as f:
                f.write(analyzer_log)
            if os.path.exists(os.path.join(RUN_DIR, "PSC_P0_VERDICT.json")):
                with open(os.path.join(RUN_DIR, "PSC_P0_VERDICT.json")) as f:
                    verdict_obj = json.load(f)
        except Exception as e:
            log("ANALYZER_EXC " + str(e))
    else:
        log("analyzer missing; writing provisional verdict")

    if verdict_obj is None:
        verdict_obj = {"schema_version": "PSC-P0-v1", "verdict": "PSC_P0_INCONCLUSIVE",
                       "reason": "analyzer produced no verdict"}

    v = verdict_obj.get("verdict", "UNKNOWN")
    report = []
    report.append("# PSC-P0 Report (Projection-Space Consensus GS, P0 falsification probe)")
    report.append("")
    report.append("STATUS: " + v)
    report.append("")
    if deviations:
        report.append("## DOCUMENTED DEVIATION FROM FROZEN PREREG")
        report.append("")
        report.append("**WARNING: The frozen prereg (PSC_P0_PREREG_v1.md) contains a malformed identity constant for `c0_extension` (63 hex chars — not a valid SHA-256, which must be 64). No file can ever match it.**")
        report.append("")
        report.append("The on-disk C0 extension was independently verified as the genuine frozen artifact by the authoritative P12 source-identity record (`P12_SOURCE_IDENTITY.json` -> `c0_lineage.ext_so.sha256`), which records the exact 64-char hash present on disk. The identity check was bound to that independent authoritative value. This is a documented correction of a prereg transcription typo, NOT a FAIL->PASS upgrade of a genuine mismatch. Full details in FROZEN_IDENTITIES.json.")
        report.append("")
        for d in deviations:
            report.append("- Identity: `%s`" % d["identity"])
            report.append("  - Prereg constant (malformed, %d chars): `%s`" % (d["prereg_constant_len"], d["prereg_constant"]))
            report.append("  - Authoritative value used (%s): `%s`" % (d["authoritative_source"], d["authoritative_value_used"]))
            report.append("  - On-disk actual: `%s` (matches authoritative: %s)" % (d["on_disk_actual"], d["on_disk_matches_authoritative"]))
            report.append("  - Reason: %s" % d["reason"])
        report.append("")
    report.append("## OBSERVED")
    report.append("- Host: " + env["hostname"] + ", GPU: " + env["gpu"] + ", torch " + env["torch"] + ", cuda " + env["cuda"])
    report.append("- Frozen identities all match: " + str(all_match))
    report.append("- N Gaussians: " + str(N) + "; cameras: " + str(len(cameras)) + " (" + str(cameras) + ")")
    report.append("- Timing (camera 19, " + str(REPS) + " reps): T_screen=%.3f ms, T_global=%.3f ms (proj=%.3f vjp=%.3f opt=%.3f), T_native=%.3f ms, f_global=%.3f, S_max=%.3f" % (T_screen, T_global, T_proj, T_vjp, T_opt, T_native, f_global, S_max))
    report.append("- S_upper: K=2 %.4f, K=4 %.4f, K=8 %.4f" % (S_upper["2"], S_upper["4"], S_upper["8"]))
    report.append("- R_int_energy (primary): B=2 %.4f, B=4 %.4f, B=8 %.4f" % (re_p["2"], re_p["4"], re_p["8"]))
    report.append("- Leave-one-out (primary, B=8): P_hold_pos=%.4f, median_C_hold=%.4f" % (loo_p["P_hold_pos"], loo_p["median_C_hold"]))
    report.append("- Diagnostic R_int_energy: B=2 %.4f, B=4 %.4f, B=8 %.4f" % (re_d["2"], re_d["4"], re_d["8"]))
    report.append("")
    report.append("## EVIDENCE")
    report.append("- Timing segments isolated with CUDA events (warmup=" + str(WARMUP) + ", reps=" + str(REPS) + ").")
    report.append("- Geometry state z=(mu_2d, vech(Sigma_2d)); exact autograd Jacobian (5 VJP passes), validated vs per-Gaussian jacobian in smoke test.")
    report.append("- Whitening W=block_diag(Sigma^-1/2, vech-whitener); R_int=||J~ pinv(J~) d_white||^2/||d_white||^2.")
    report.append("- Cohort: top-4096 by accumulated |means2d.absgrad| over 8 frozen views (primary); 4096 uniform visible (diagnostic).")
    report.append("")
    report.append("## HYPOTHESIS")
    report.append("- H1: image-rate 3D consistency is required (PSC will not promote).")
    report.append("- H2: multiple high-frequency projected updates can be amortized behind lower-frequency multi-view 3D sync (PSC will promote).")
    report.append("")
    report.append("## TEST")
    report.append("- Test A (upper bound): S_upper(K)=(T_screen+T_global)/(T_screen+T_global/K).")
    report.append("- Test B (integrability): R_int_energy for B in {2,4,8}.")
    report.append("- Test C (leave-one-out): cosine consistency, P_hold_pos.")
    report.append("")
    report.append("## CONCLUSION")
    report.append("- Mechanical verdict (owned by psc_p0_analyze.py): " + v)
    report.append("- Analyzer stdout:")
    report.append("```")
    report.append(analyzer_log)
    report.append("```")
    report.append("")
    report.append("Note: P0 is a falsification probe only. No production implementation is authorized unless verdict is STRONG_PROMOTE; even then, P1 is not started here.")
    write_text("PSC_P0_REPORT.md", "\n".join(report) + "\n")

    finalize_manifest()
    # Release the single-instance lock (fcntl locks also auto-release on process exit).
    try:
        if _lock_fp is not None:
            fcntl.flock(_lock_fp, fcntl.LOCK_UN)
            _lock_fp.close()
    except Exception:
        pass
    log("=== PSC-P0 probe complete verdict=" + v + " elapsed=%.1fs ===" % (time.time() - t_start))
    log(json.dumps({"verdict": v, "S_upper_8": S_upper["8"], "R_int_energy_8": re_p["8"],
                    "P_hold_pos_8": loo_p["P_hold_pos"], "median_C_hold_8": loo_p["median_C_hold"]}))
    return

def finalize_manifest():
    entries = []
    for name in sorted(os.listdir(RUN_DIR)):
        p = os.path.join(RUN_DIR, name)
        if os.path.isfile(p):
            entries.append(name + "  " + sha256_file(p))
    man = "\n".join(entries) + "\n"
    write_text("MANIFEST.sha256", man)
    log("MANIFEST written with " + str(len(entries)) + " entries")

if __name__ == "__main__":
    main()
