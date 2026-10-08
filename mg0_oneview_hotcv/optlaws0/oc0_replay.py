#!/usr/bin/env python3
"""OPTLAWS0 / OC0 — exact native renderer replay engine (vectorized).

Reimplements, in vectorized torch, the EXACT per-pixel loop of the production
gsplat forward kernel `rasterize_to_pixels_3dgs` that the frozen c0 render
branch already ran.  Nothing here re-renders with a different code path: the
inputs are the tensors the native forward ALREADY computed and published
(see oc0_patch_renderer.py), so the replay is a pure re-derivation of the
native loop, used only to extract per-Gaussian renderer summaries.

Native loop being replayed (verified against three independent gsplat sources
and against the frozen worktree's CUDA binary):

    px = j + 0.5 ; py = i + 0.5                    (tile-local pixel centre)
    for l in tile's front-to-back isect list:
        sigma = 0.5*(c0*dx^2 + c2*dy^2) + c1*dx*dy
        alpha = min(0.999, opac*exp(-sigma))
        if sigma < 0 or alpha < 1/255:  continue          (not a hit)
        next_T = T*(1-alpha)
        if next_T <= 1e-4:  done = True ; break           (THIS one excluded)
        out += color*alpha*T ; cur_idx = flat_isect_index ; T = next_T
    render_alpha = 1 - T ; render_color = out           (bg is None -> 0)

Because the running transmittance
    T_l = prod_{hit u <= l} (1 - alpha_u)
is non-increasing in l, the accepted set is exactly the PREFIX of the hit list
with T_l > 1e-4, so the per-pixel state is a running product plus a segmented
prefix cut-off -- no stateful per-pixel loop is needed.

The product is formed in the SAME multiplicative form and the SAME float32
precision the kernel uses (``T = T * (1 - alpha)``), rather than via a
log-domain cumsum: the acceptance test is the kernel's own comparison
``T > 1e-4``, so the cut-off lands on the same gaussian except for float32
rounding of ``expf`` and of the product itself.

Everything is computed in float32 on the same device as the captured tensors.
`last_ids` is NOT used to build the replay; it is used only as an independent
cross-check of the last accepted isect index per pixel.

Diagnostic only: no training, no renderer change, no threshold tuning.
"""
import math

import torch

import oc0_common as K


def _as_f32(t):
    return t.detach().float()


def replay_camera(captured, visible_ids, selected_ids, tile_size=16,
                  force_all_tiles=False, want_rgb=False,
                  l_chunk=128, tile_batch=64, return_pixel_buffers=None):
    """Replay one camera's native forward and collect per-Gaussian summaries.

    Parameters
    ----------
    captured : tuple
        The exact tuple published by the instrumented native forward:
        (means2d, conics, colors_eval, opacities_bc, isect_offsets,
         flatten_ids, render_alphas, last_ids, radii, depth_acc|None).
    visible_ids : LongTensor [N_visible]
        Master row index of each visible slot (the gatherless producer input).
    selected_ids : LongTensor [ng]
        Master rows of the frozen OC0 selection, in frozen order.  Outputs are
        indexed by position in this tensor (-1 sparse check is internal).
    force_all_tiles : bool
        If True, every non-empty tile is replayed (required for whole-image
        audio/RGB parity).  If False, only tiles touched by a selected row are
        replayed (required only for per-Gaussian summaries).
    want_rgb : bool
        Also accumulate the composited RGB image (only 3-channel renders).
    return_pixel_buffers : bool or None
        Defaults to ``force_all_tiles``.

    Returns
    -------
    dict with per-Gaussian aggregates over the replay ("agg"), optional
    whole-image buffers ("alpha_full", "rgb_full", "processed"), accumulated
    accepted counts ("acc_total"), and the independent cross-check errors.
    """
    if return_pixel_buffers is None:
        return_pixel_buffers = bool(force_all_tiles)

    (means2d, conics, colors, opac, isect_offsets, flatten_ids,
     render_alphas, last_ids, radii, depth_acc) = captured
    dev = means2d.device
    render_alphas = _as_f32(render_alphas)
    H, W = int(render_alphas.shape[-2]), int(render_alphas.shape[-1])
    th, tw = int(isect_offsets.shape[-2]), int(isect_offsets.shape[-1])
    nvis = int(means2d.shape[0])
    assert means2d.shape == (nvis, 2), tuple(means2d.shape)
    assert conics.shape == (nvis, 3), tuple(conics.shape)
    assert opac.shape[0] == nvis, tuple(opac.shape)
    assert int(visible_ids.numel()) == nvis, (visible_ids.numel(), nvis)
    assert th == int(math.ceil(H / tile_size)), (th, H)
    assert tw == int(math.ceil(W / tile_size)), (tw, W)
    if want_rgb:
        assert int(colors.shape[-1]) == 3, tuple(colors.shape)

    means2d = _as_f32(means2d).contiguous()
    conics = _as_f32(conics).contiguous()
    colors = _as_f32(colors).contiguous()
    opac = _as_f32(opac).contiguous()
    radii = _as_f32(radii).reshape(-1, 2).contiguous()
    assert radii.shape[0] == nvis, tuple(radii.shape)

    fi = flatten_ids.detach().to(torch.int64)
    n = int(fi.numel())
    T = th * tw
    off = isect_offsets.detach().reshape(-1).to(torch.int64)
    assert off.numel() == T
    counts = torch.cat(
        [off[1:], torch.tensor([n], device=dev, dtype=torch.int64)]) - off
    assert int(counts.sum()) == n
    assert int(counts.min()) >= 0

    # ---- master row -> position in the frozen selection ---------------------
    # The LUT must cover BOTH index sets it is used with: the visible slots
    # (scan) and the frozen selection rows (scatter).  A selected row need not
    # be visible from this camera, so its master id can exceed
    # max(visible_ids) -- sizing the LUT by the visible max alone made the
    # scatter go out of bounds.
    sel = selected_ids.detach().to(dev, torch.int64)
    ng = int(sel.numel())
    n_lut = int(max(visible_ids.detach().max().item(),
                    sel.max().item())) + 1
    lut = torch.full((n_lut,), -1, dtype=torch.int64, device=dev)
    lut[sel] = torch.arange(ng, device=dev, dtype=torch.int64)
    slot_sel = lut[visible_ids.detach().to(dev, torch.int64)]   # [nvis]
    assert int(((slot_sel >= 0).sum())) <= ng
    isect_sel = slot_sel[fi] >= 0                               # [n]

    # ---- which tiles to replay ---------------------------------------------
    if force_all_tiles:
        tiles = torch.nonzero(counts > 0, as_tuple=False).flatten()
    else:
        if bool(isect_sel.any()):
            tile_of_entry = torch.repeat_interleave(
                torch.arange(T, device=dev, dtype=torch.int64), counts)
            tiles = torch.unique(tile_of_entry[isect_sel])
        else:
            tiles = torch.empty(0, dtype=torch.int64, device=dev)

    npix_tile = tile_size * tile_size
    pid_r = (torch.arange(npix_tile, device=dev) // tile_size)   # [P]
    pid_c = (torch.arange(npix_tile, device=dev) % tile_size)    # [P]

    acc_buf = torch.zeros(H * W, dtype=torch.int32, device=dev)
    last_buf = torch.full((H * W,), -1, dtype=torch.int64, device=dev)
    touch_buf = torch.zeros(H * W, dtype=torch.bool, device=dev)
    alpha_buf = torch.zeros(H * W, dtype=torch.float32, device=dev)
    rgb_buf = (torch.zeros(H * W, 3, dtype=torch.float32, device=dev)
               if (return_pixel_buffers and want_rgb) else None)

    pg, pp, pa, ptb, pd, ppr = [], [], [], [], [], []

    n_processed_tiles = 0
    n_cand = 0
    for b0 in range(0, int(tiles.numel()), tile_batch):
        tb = tiles[b0:b0 + tile_batch]
        B = int(tb.numel())
        cnt = counts[tb]                                        # [B]
        Lmax = int(cnt.max().item())
        lpos = torch.arange(Lmax, device=dev, dtype=torch.int64)
        eidx = (off[tb][:, None] + lpos[None, :]).clamp_(max=max(n - 1, 0))
        evalid = lpos[None, :] < cnt[:, None]                    # [B,Lmax]

        ti = tb // tw
        tj = tb % tw
        rr = ti[:, None] * tile_size + pid_r[None, :]            # [B,P]
        cc = tj[:, None] * tile_size + pid_c[None, :]
        valid_pix = (rr < H) & (cc < W)
        pf = rr * W + cc                                         # [B,P]
        px = cc.to(torch.float32) + 0.5
        py = rr.to(torch.float32) + 0.5

        dden = (cnt - 1).clamp_min(1).to(torch.float32)[:, None]

        prev_T = torch.ones(B, npix_tile, dtype=torch.float32, device=dev)
        prev_pass = torch.zeros(B, npix_tile, dtype=torch.float32, device=dev)
        acc_cnt = torch.zeros(B, npix_tile, dtype=torch.int32, device=dev)
        last_hit = torch.full((B, npix_tile), -1, dtype=torch.int64, device=dev)
        last_T = torch.full((B, npix_tile), float("inf"),
                            dtype=torch.float32, device=dev)
        rgb_acc = (torch.zeros(B, npix_tile, 3, dtype=torch.float32, device=dev)
                   if (return_pixel_buffers and want_rgb) else None)

        for s in range(0, Lmax, l_chunk):
            e = min(s + l_chunk, Lmax)
            ids = eidx[:, s:e]                                   # [B,Cp]
            vm = evalid[:, s:e]                                  # [B,Cp]
            Cp = e - s
            slot = fi[ids]                                       # [B,Cp]
            mg = means2d[slot]                                   # [B,Cp,2]
            cn = conics[slot]
            op = opac[slot]                                      # [B,Cp]

            dx = mg[..., 0:1] - px[:, None, :]
            dy = mg[..., 1:2] - py[:, None, :]
            sigma = (0.5 * (cn[..., 0:1] * dx * dx + cn[..., 2:3] * dy * dy)
                     + cn[..., 1:2] * dx * dy)
            alpha = torch.clamp(op[:, :, None] * torch.exp(-sigma),
                        max=K.MAX_ALPHA)
            hit = ((sigma >= 0.0) & (alpha >= K.ALPHA_THRESHOLD)
                   & vm[:, :, None] & valid_pix[:, None, :])      # [B,Cp,P]
            n_cand += int(hit.sum().item())

            # Same arithmetic the kernel performs: fac = (1 - alpha) on hits,
            # exactly 1.0 on skipped entries, and T is a running product.
            fac = torch.where(hit, 1.0 - alpha, torch.ones_like(alpha))
            T_after = prev_T[:, None, :] * torch.cumprod(fac, dim=1)
            # exclusive product == the kernel's T immediately before this entry
            T_before = torch.cat([prev_T[:, None, :], T_after[:, :-1, :]], dim=1)
            passf = hit.to(torch.float32)
            cum_pass = prev_pass[:, None, :] + torch.cumsum(passf, dim=1)
            prior = (cum_pass - passf)                            # [B,Cp,P]

            accepted = hit & (T_after > K.TRANSMITTANCE_THRESHOLD)

            acc_cnt += accepted.sum(dim=1).to(torch.int32)
            cand_T = torch.where(accepted, T_after,
                                 torch.full_like(T_after, float("inf")))
            last_T = torch.minimum(last_T, cand_T.amin(dim=1))
            cand_l = torch.where(accepted,
                                 ids[:, :, None].expand_as(accepted), -1)
            last_hit = torch.maximum(last_hit, cand_l.amax(dim=1))

            sel_idx = slot_sel[slot]                              # [B,Cp]
            take = accepted & (sel_idx[:, :, None] >= 0)
            if bool(take.any()):
                pg.append(sel_idx[:, :, None].expand_as(take)[take]
                          .to(torch.int32).cpu())
                pp.append(pf[:, None, :].expand_as(take)[take]
                          .to(torch.int32).cpu())
                pa.append(alpha[take].cpu())
                ptb.append(T_before[take].cpu())
                ppr.append(prior[take].to(torch.int32).cpu())
                lpos_c = lpos[s:e].to(torch.float32)[None, :] / dden   # [B,Cp]
                pd.append(lpos_c[:, :, None].expand(B, Cp, npix_tile)[take]
                          .cpu())

            if rgb_acc is not None:
                w = torch.where(accepted, alpha * T_before,
                                torch.zeros_like(alpha))
                rgb_acc += torch.einsum("bcp,bcd->bpd", w, colors[slot])

            prev_T = T_after[:, -1, :].contiguous()
            prev_pass = cum_pass[:, -1, :].contiguous()

        T_final = torch.where(torch.isfinite(last_T), last_T,
                              torch.ones_like(last_T))
        alpha_tile = 1.0 - T_final

        m = valid_pix
        acc_buf[pf[m]] += acc_cnt[m]
        touch_buf[pf[m]] = True
        lm = m & (last_hit >= 0)
        last_buf[pf[lm]] = last_hit[lm]
        alpha_buf[pf[m]] = alpha_tile[m]
        if rgb_buf is not None:
            rgb_buf[pf[m]] = rgb_acc[m]
        n_processed_tiles += B

    out = {
        "n_gaussians": ng,
        "n_visible_slots": nvis,
        "n_isects": n,
        "n_processed_tiles": n_processed_tiles,
        "n_candidate_pairs": n_cand,
        "pair_sel_index": (torch.cat(pg) if pg else
                           torch.empty(0, dtype=torch.int32)),
        "pair_pixel": (torch.cat(pp) if pp else
                       torch.empty(0, dtype=torch.int32)),
        "pair_alpha": (torch.cat(pa) if pa else torch.empty(0)),
        "pair_T_before": (torch.cat(ptb) if ptb else torch.empty(0)),
        "pair_prior": (torch.cat(ppr) if ppr else
                       torch.empty(0, dtype=torch.int32)),
        "pair_depth_pct": (torch.cat(pd) if pd else torch.empty(0)),
        "acc_total": acc_buf,
        "last_isect": last_buf,
        "radius_x": radii[:, 0].cpu(),
        "radius_y": radii[:, 1].cpu(),
        "slot_sel_index": slot_sel.cpu(),
    }

    # ---- independent cross-checks against the native outputs ----------------
    native_alpha = render_alphas.reshape(-1)
    if return_pixel_buffers:
        # every in-bounds pixel is covered: non-empty tiles are replayed and
        # empty tiles keep the exact native value 1 - T = 0.
        processed = torch.ones(H * W, dtype=torch.bool, device=dev)
    else:
        processed = touch_buf
    aerr = (alpha_buf - native_alpha).abs() * processed
    out["max_abs_alpha_err"] = float(aerr.max().item())
    out["mean_abs_alpha_err"] = float(aerr.sum().item()
                                      / max(int(processed.sum().item()), 1))

    native_last = last_ids.detach().reshape(-1).to(torch.int64)
    cmp_mask = last_buf >= 0
    if bool(cmp_mask.any()):
        out["max_abs_lastid_err"] = float(
            (last_buf[cmp_mask] - native_last[cmp_mask]).abs().max().item())
        out["n_lastid_compared"] = int(cmp_mask.sum().item())
        out["n_lastid_mismatch"] = int(
            (last_buf[cmp_mask] != native_last[cmp_mask]).sum().item())
    else:
        out["max_abs_lastid_err"] = None
        out["n_lastid_compared"] = 0
        out["n_lastid_mismatch"] = 0

    out["alpha_full"] = (alpha_buf.reshape(H, W) if return_pixel_buffers
                         else None)
    out["rgb_full"] = (rgb_buf.reshape(H, W, 3) if rgb_buf is not None
                       else None)
    out["processed"] = processed.reshape(H, W) if return_pixel_buffers else None
    return out


def finalize_aggregates(res, ng, acc_total=None):
    """Reduce the raw accepted-hit pairs to the frozen per-Gaussian summaries.

    Returns torch float64 tensors of length ng:
      accepted_hits, mean_alpha, mean_T_before, p10_T_before, p50_T_before,
      mean_depth_percentile, mean_prior_contrib_count,
      mean_remaining_contrib_count
    plus the pair counts actually used.  Missing quantities (no accepted hit)
    are NaN -- never invented.
    """
    import numpy as np

    # The replay buffers live on the CUDA device; `.numpy()` alone raises
    # TypeError on a device tensor, so go through an explicit host copy.
    def _np(t):
        return t.detach().cpu().numpy()

    g = _np(res["pair_sel_index"])
    p = _np(res["pair_pixel"])
    a = _np(res["pair_alpha"]).astype(np.float64)
    tb = _np(res["pair_T_before"]).astype(np.float64)
    pr = _np(res["pair_prior"]).astype(np.int64)
    dp = _np(res["pair_depth_pct"]).astype(np.float64)
    acc = _np(res["acc_total"] if acc_total is None else acc_total)

    hits = np.bincount(g, minlength=ng).astype(np.int64)
    sa = np.bincount(g, weights=a, minlength=ng)
    st = np.bincount(g, weights=tb, minlength=ng)
    sd = np.bincount(g, weights=dp, minlength=ng)
    sp = np.bincount(g, weights=pr.astype(np.float64), minlength=ng)
    rem = acc[p.astype(np.int64)].astype(np.float64) - pr - 1.0
    sr = np.bincount(g, weights=rem, minlength=ng)

    nz = hits > 0
    den = np.where(nz, hits, 1).astype(np.float64)
    mean_a = np.where(nz, sa / den, np.nan)
    mean_t = np.where(nz, st / den, np.nan)
    mean_d = np.where(nz, sd / den, np.nan)
    mean_p = np.where(nz, sp / den, np.nan)
    mean_r = np.where(nz, sr / den, np.nan)

    p10 = np.full(ng, np.nan)
    p50 = np.full(ng, np.nan)
    if len(g):
        order = np.lexsort((tb, g))
        gs, ts = g[order], tb[order]
        starts = np.searchsorted(gs, np.arange(ng), side="left")
        ends = np.searchsorted(gs, np.arange(ng), side="right")
        for j in np.nonzero(ends > starts)[0]:
            vals = ts[starts[j]:ends[j]]
            p10[j] = np.percentile(vals, 10)
            p50[j] = np.percentile(vals, 50)

    t = torch.as_tensor
    return {
        "accepted_hits": t(hits, dtype=torch.int64),
        "mean_alpha": t(mean_a),
        "mean_T_before": t(mean_t),
        "p10_T_before": t(p10),
        "p50_T_before": t(p50),
        "mean_depth_percentile": t(mean_d),
        "mean_prior_contrib_count": t(mean_p),
        "mean_remaining_contrib_count": t(mean_r),
        "n_pair_entries": int(g.size),
    }