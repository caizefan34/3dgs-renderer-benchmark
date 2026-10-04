

# ---------------------------------------------------------------------------
# 2D covariance projection matching gsplat 1.5.3 (perspective Jacobian,
# eps2d blur). Convention recorded; eps2d must match the native kernel used.
# ---------------------------------------------------------------------------

def _quat_to_R(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=1).reshape(-1, 3, 3)
    return R


def project_gaussians_2d(scene: Scene, camera: Camera, eps2d: float = 0.1):
    """Project 3D Gaussians to 2D screen space (gsplat convention)."""
    R = _quat_to_R(scene.rotation)
    S = torch.exp(scene.scaling)
    M = R @ S.unsqueeze(-1)
    Sigma = M @ M.transpose(1, 2)
    Rcw = camera.cam_from_world[:3, :3]
    tcw = camera.cam_from_world[:3, 3]
    t = (Rcw @ scene.xyz.unsqueeze(-1)).squeeze(-1) + tcw
    depth = t[:, 2]
    Sigma_c = Rcw @ Sigma @ Rcw.transpose(0, 1)
    fx, fy = camera.fx, camera.fy
    tx, ty, tz = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    tz2 = tz * tz
    J = torch.zeros((t.shape[0], 2, 3), dtype=torch.float32, device=t.device)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -fx * tx / tz2
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -fy * ty / tz2
    Sigma2d = J @ Sigma_c @ J.transpose(1, 2)
    Sigma2d[:, 0, 0] += eps2d
    Sigma2d[:, 1, 1] += eps2d
    a = Sigma2d[:, 0, 0]; b = Sigma2d[:, 0, 1]; c = Sigma2d[:, 1, 1]
    det = torch.clamp(a * c - b * b, min=1e-12)
    conic = torch.stack([c / det, -b / det, a / det], dim=-1)
    mu2d = torch.empty((t.shape[0], 2), dtype=torch.float32, device=t.device)
    mu2d[:, 0] = fx * tx / tz + camera.cx
    mu2d[:, 1] = fy * ty / tz + camera.cy
    mid = 0.5 * (a + c)
    disc = torch.clamp(mid * mid - det, min=0.0)
    lam = mid + torch.sqrt(disc)
    radius = torch.ceil(3.0 * torch.sqrt(lam)).int()
    valid = (depth > 0.05) & (radius > 0)
    radii = torch.where(valid, radius, torch.zeros_like(radius))
    return mu2d, conic, depth, radii


# ---------------------------------------------------------------------------
# native_alpha: alpha evaluator identical in form to the gsplat CUDA kernel.
# ---------------------------------------------------------------------------

ALPHA_CAP = 0.999
ALPHA_CUTOFF = 1.0 / 255.0


def alpha_from_q(q, opacity):
    a = torch.clamp(opacity * torch.exp(-0.5 * q), max=ALPHA_CAP)
    return torch.where(a >= ALPHA_CUTOFF, a, torch.zeros_like(a))


def native_alpha(mu2d, conic, opacity, pixel_xy):
    """Alpha for N occurrences at P pixel centers -> [N,P]."""
    N = mu2d.shape[0]; P = pixel_xy.shape[0]
    dx = pixel_xy[:, 0].unsqueeze(0) - mu2d[:, 0].unsqueeze(1)
    dy = pixel_xy[:, 1].unsqueeze(0) - mu2d[:, 1].unsqueeze(1)
    a = conic[:, 0].unsqueeze(1); b = conic[:, 1].unsqueeze(1); c = conic[:, 2].unsqueeze(1)
    q = a * dx * dx + 2 * b * dx * dy + c * dy * dy
    op = opacity.reshape(-1)
    if op.shape[0] != N:
        op = op.expand(N)
    return alpha_from_q(q, op.unsqueeze(1).expand(N, P).reshape(-1)).reshape(N, P)


# ---------------------------------------------------------------------------
# project_and_sort: tile binning + stable depth sort
# ---------------------------------------------------------------------------

def project_and_sort(scene: Scene, camera: Camera, tile_size: int) -> TileLists:
    mu2d, conic, depth, radii = project_gaussians_2d(scene, camera)
    op_act = opacity_activation(scene.opacity)
    W, H = camera.width, camera.height
    tiles_x = (W + tile_size - 1) // tile_size
    tiles_y = (H + tile_size - 1) // tile_size
    tile_of = {}
    ids = torch.nonzero(radii > 0, as_tuple=False).squeeze(-1)
    mu2d_np = mu2d.detach().cpu().numpy()
    radii_np = radii.detach().cpu().numpy()
    for gid in ids.tolist():
        g = int(gid); r = int(radii_np[g])
        cx, cy = float(mu2d_np[g, 0]), float(mu2d_np[g, 1])
        x0 = max(0, int(math.floor(cx - r))); x1 = min(W - 1, int(math.floor(cx + r)))
        y0 = max(0, int(math.floor(cy - r))); y1 = min(H - 1, int(math.floor(cy + r)))
        if x1 < x0 or y1 < y0:
            continue
        for ty in range(y0 // tile_size, y1 // tile_size + 1):
            for tx in range(x0 // tile_size, x1 // tile_size + 1):
                tile_of.setdefault((ty, tx), []).append(g)
    tile_keys = sorted(tile_of.keys())
    tile_gids = []; tile_deps = []
    depth_np = depth.detach().cpu().numpy()
    for tid in tile_keys:
        gids = tile_of[tid]
        deps = [depth_np[g] for g in gids]
        order = np.argsort(deps, kind="stable")
        tile_gids.append(np.array([gids[i] for i in order], dtype=np.int64))
        tile_deps.append(np.array([deps[i] for i in order], dtype=np.float64))
    return TileLists(tile_size=tile_size, width=W, height=H,
                     tiles_x=tiles_x, tiles_y=tiles_y,
                     tile_gaussian_ids=tile_gids, tile_depths=tile_deps,
                     camera=camera,
                     meta={"mu2d": mu2d, "conic": conic, "radii": radii,
                           "depth": depth, "opacity": op_act, "tile_keys": tile_keys})


# ---------------------------------------------------------------------------
# build_bounds: R0 enumerated per-occurrence bounds + R/D routing (gamma=2)
# ---------------------------------------------------------------------------

def build_bounds(tile_lists: TileLists, mode: str = "R0_ENUMERATED_NATIVE") -> BoundsAndRouting:
    """R0: enumerate every in-tile pixel center per occurrence, call
    native_alpha, take discrete min/max, nextafter-outward, clamp to
    [0,cap], route R if l>0 and u/l<=gamma else D. Occurrences with l=0
    (support/cutoff boundary) go to D. Requires activated opacity in meta.
    """
    assert mode.startswith("R0"), "only R0 enumeration implemented in this build"
    gamma = 2.0
    ts = tile_lists.tile_size
    mu2d = tile_lists.meta["mu2d"]; conic = tile_lists.meta["conic"]
    radii = tile_lists.meta["radii"]; opacity = tile_lists.meta["opacity"]
    occ_g = []; occ_t = []; occ_xy = []
    lo_list = []; hi_list = []
    for tindex, (ty, tx) in enumerate(tile_lists.meta["tile_keys"]):
        gids = tile_lists.tile_gaussian_ids[tindex]
        px0 = tx * ts; py0 = ty * ts
        px1 = min(tile_lists.width, px0 + ts); py1 = min(tile_lists.height, py0 + ts)
        xs = np.arange(px0, px1); ys = np.arange(py0, py1)
        grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
        pix = torch.tensor(grid, dtype=torch.float32, device=mu2d.device)
        if gids.size == 0:
            continue
        m = mu2d[gids]; cc = conic[gids]; op = opacity[gids]
        alphas = native_alpha(m, cc, op, pix).detach().cpu().numpy()  # [g, P]
        with np.errstate(invalid="ignore"):
            lo = alphas.min(axis=1); hi = alphas.max(axis=1)
        occ_g.append(gids); occ_t.append(np.full(gids.shape, tindex, dtype=np.int64))
        occ_xy.append(grid)
        lo_list.append(lo); hi_list.append(hi)
    gaussian_ids = np.concatenate(occ_g) if occ_g else np.zeros(0, np.int64)
    tile_ids = np.concatenate(occ_t) if occ_t else np.zeros(0, np.int64)
    pixel_xy = np.concatenate(occ_xy, axis=0) if occ_xy else np.zeros((0, 2))
    lo = np.concatenate(lo_list) if lo_list else np.zeros(0)
    hi = np.concatenate(hi_list) if hi_list else np.zeros(0)
    lo = np.nextafter(lo, -np.inf).astype(np.float64)
    hi = np.nextafter(hi, np.inf).astype(np.float64)
    lo = np.clip(lo, 0.0, ALPHA_CAP); hi = np.clip(hi, 0.0, ALPHA_CAP)
    is_R = (lo > 0.0) & (hi / np.maximum(lo, 1e-300) <= gamma)
    return BoundsAndRouting(mode=mode, gaussian_ids=gaussian_ids, tile_ids=tile_ids,
                            pixel_xy=pixel_xy, alpha_lower=lo, alpha_upper=hi,
                            is_R=is_R, meta={"gamma": gamma, "tile_size": ts})
